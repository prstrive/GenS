import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F

from .modules.feature_network_mnasnet import FeatureNetwork
from .modules.volume import Volume
from .modules.reg_network import RegNetwork
from .modules.implicit_surface import ImplicitSurface


class GenS(nn.Module):
    def __init__(self, confs):
        super(GenS, self).__init__()
        
        self.has_vol = confs.get_bool("has_vol", default=False)
        
        if not self.has_vol:
            self.feature_network = FeatureNetwork(confs["feature_network"])
            self.volume = Volume(confs["volume"])
            self.reg_network = RegNetwork(confs["reg_network"])            
            self.match_feature_network = FeatureNetwork(confs["feature_network"])
            for param in self.match_feature_network.parameters():
                param.requires_grad = False
        else:
            self.volumes = nn.ParameterList(torch.empty(0))
            self.mask_volmes = nn.ParameterList(torch.empty(0))
            self.features = nn.ParameterList(torch.empty(0))
            
        self.implicit_surface = ImplicitSurface(confs["implicit_surface"])
        
    def get_optim_params(self, lr_confs):
        mlp_params_to_train = list(self.implicit_surface.parameters())
        grad_vars = [{'params': mlp_params_to_train, 'lr': lr_confs["mlp_lr"]}]
        # grad_vars = [{'params': list(self.implicit_surface.sdf_network.parameters())+list(self.implicit_surface.deviation_network.parameters()), 'lr': lr_confs["mlp_lr"]}]
        # grad_vars.append({'params': list(self.implicit_surface.color_network.parameters()), 'lr': 5e-4})
        if not self.has_vol:
            feat_params_to_train = list(self.feature_network.parameters()) + list(self.reg_network.parameters()) #+ list(self.volume.parameters())
            grad_vars.append({'params': feat_params_to_train, 'lr': lr_confs["feat_lr"]})
        else:
            for volume_param, v_lr in zip(self.volumes, lr_confs["vol_lr"]):
                grad_vars.append({'params': volume_param, 'lr': v_lr})
        return grad_vars
    
    def load_params_vol(self, path, device):
        ckpt = torch.load(path)
        model = ckpt["model"]
        self.volumes = model["volumes"].to(device)
        self.mask_volmes = model["mask_volmes"].to(device)
        self.features = model["features"].to(device)
        self.implicit_surface.load_state_dict(model["implicit_surface"])
        self.has_vol = True
        
    def get_params_vol(self):
        params = {
            "volumes": self.volumes,
            "mask_volmes": self.mask_volmes,
            "features": self.features,
            "implicit_surface": self.implicit_surface.state_dict()
        }
        return params
    
    def init_volumes(self, ipts):
        imgs = ipts["imgs"]  # (nv, 3, h, w)
        intrs = ipts["intrs"]   # (nv, 4, 4)
        c2ws = ipts["c2ws"] # (nv, 4, 4)
        
        with torch.no_grad():
            features = self.feature_network(imgs)   # coarse to fine
            volumes, mask_volmes = self.volume.agg_mean_var(features, intrs, c2ws, min_vis_view=1)
            volumes = self.reg_network(volumes)
            
        # mask_volmes = self.filter_volume(volumes, mask_volmes)
        
        self.volumes = nn.ParameterList([nn.Parameter(volume.detach(), requires_grad=True) for volume in volumes])
        self.mask_volmes = nn.ParameterList([nn.Parameter(volume.detach(), requires_grad=False) for volume in mask_volmes])
        self.features = nn.ParameterList([nn.Parameter(feat.detach(), requires_grad=False) for feat in features])
        # self.features = []
        # ori_hw = np.array([1200, 1600])
        # for i in range(len(features)):
        #     cur_hw = ori_hw * 0.5 ** i
        #     feat = F.interpolate(features[i], size=(int(cur_hw[0]), int(cur_hw[1])), mode="bilinear", align_corners=True)
        #     self.features.append(feat)
            
        self.has_vol = True
    
    @torch.no_grad()  
    def filter_volume(self, volumes, mask_volmes, thresh=0.1):
        print("Filtering sdf volume...")
        
        # projector and feather net no flip
        _, c, w, h, d = volumes[0].shape
        # w, h, d = 256, 256, 256
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        z = torch.linspace(-1, 1, d)
        coord_volume = torch.stack(torch.meshgrid([z, y, x])[::-1], dim=-1).view(-1, 3).type_as(volumes[0]) # (dhw, 3)
        chunk_size = 128*128*128
        pts = coord_volume.split(chunk_size)
        
        # sdf_all = []
        mask_all = []
        for pts_part in pts:
            sdf = self.implicit_surface.sdf_network.sdf(pts_part, volumes, None)
            # sdf_all.append(sdf)
            mask_all.append((sdf.abs() < thresh).float())
            del sdf
        
        # sdf_volume = torch.cat(sdf_all, dim=0).view(1, 1, w, h, d)
        mask = torch.cat(mask_all, dim=0).view(1, 1, d, h, w)
        coord_norm = torch.linalg.norm(coord_volume, ord=2, dim=-1, keepdim=False).reshape(1, 1, d, h, w)
        mask = mask * (coord_norm < 1).float()
        print("Survival ratio:", mask.mean())
        mask = F.max_pool3d(mask, 3, 1, 1)
        print("Survival ratio after dilation:", mask.mean())
        mask = mask.permute(0, 1, 4, 3, 2)
        
        for i in range(len(mask_volmes)):
            mask_volmes[i] = mask_volmes[i] * mask
            mask = F.interpolate(mask, scale_factor=0.5, mode="nearest")
            
        return mask_volmes
        
    def forward(self, mode, ipts, cos_anneal_ratio=1.0, step=None):
        
        if not self.has_vol:
            imgs = ipts["imgs"]  # (nv, 3, h, w)
            intrs = ipts["intrs"]   # (nv, 4, 4)
            c2ws = ipts["c2ws"] # (nv, 4, 4)
            
            features = self.feature_network(imgs)   # coarse to fine
            
            if step is not None and step%5 == 0:
                print("load image feature ckpt")
                last_state = self.feature_network.state_dict()
                self.match_feature_network.load_state_dict(last_state, strict=True)
                for param in self.match_feature_network.parameters():
                    param.requires_grad = False
            
            with torch.no_grad():
                match_features = self.match_feature_network(imgs)
            
            volumes, mask_volmes = self.volume.agg_mean_var(features, intrs, c2ws)
            
            volumes = self.reg_network(volumes)
        
        else:
            view_ids = ipts["view_ids"] if mode != "val" else list(range(ipts["imgs"].shape[0]))  # list of int
            
            volumes = self.volumes
            mask_volmes = self.mask_volmes
            features = [feat[view_ids] for feat in self.features]
            match_features = [feat[view_ids] for feat in self.features]

        outputs = self.implicit_surface(mode, ipts, volumes, mask_volmes, features, match_features, cos_anneal_ratio, step)
        
        return outputs