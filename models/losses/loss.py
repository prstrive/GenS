import torch
import torch.nn as nn
import torch.nn.functional as F

from .ncc import compute_LNCC


class Loss(nn.Module):
    def __init__(self, confs):
        super(Loss, self).__init__()
        
        self.color_weight = confs.get_float('color_weight')
        self.sparse_scale_factor = confs.get_float('sparse_scale_factor')
        self.sparse_weight = confs.get_float('sparse_weight')
        self.igr_weight = confs.get_float("igr_weight")
        self.mfc_weight = confs.get_float("mfc_weight")
        self.smooth_weight = confs.get_float("smooth_weight")
        self.tv_weight = confs.get_float('tv_weight')
        self.depth_weight = confs.get_float('depth_weight', default=0.0)
        self.pseudo_sdf_weight = confs.get_float('pseudo_sdf_weight', default=0.0)
        self.pseudo_depth_weight = confs.get_float('pseudo_depth_weight', default=0.0)
        
    def forward(self, preds, targets, step=None):
        valid_mask = preds['valid_mask']
        color_loss = F.l1_loss(preds["color_fine"], targets["color"], reduction='none')
        color_loss = (color_loss * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-5)
        
        eikonal_loss = preds['gradient_error'].mean() #+ preds["auxi_gradient_error"].mean()
        
        sparse_loss = torch.exp(-torch.abs(preds["sparse_sdf"]) * self.sparse_scale_factor).mean()
        
        smooth_loss = preds["smooth_error"].mean()
        
        tv_loss = preds["tv_reg"].mean()
                
        ncc = compute_LNCC(preds["ref_gray_val"], preds["sampled_gray_val"])
        ncc_mask = valid_mask * preds["mid_inside_sphere"]
        mfc_loss = 0.5 * ((ncc * ncc_mask).sum(dim=0) / (ncc_mask.sum(dim=0) + 1e-8)).squeeze(-1)
        
        if "pseudo_sdf" in preds:
            pseudo_sdf_loss = torch.abs(preds["pseudo_sdf"]).mean()
        else:
            pseudo_sdf_loss = torch.tensor(0.0).type_as(mfc_loss)
        
        if "pseudo_depth" in targets:
            pseudo_depth_loss = ((preds["render_depth"] - targets["pseudo_depth"]).abs() * (targets["pseudo_depth"] > 0).float()).sum() / ((targets["pseudo_depth"] > 0).float().sum() + 1e-8)
        else:
            pseudo_depth_loss = torch.tensor(0.0).type_as(mfc_loss)
        
        if "depth" in targets:
            depth_loss = ((preds["render_depth"] - targets["depth"]).abs() * (targets["depth"]>0).float()).sum() / ((targets["depth"]>0).float().sum() + 1e-8)
        else:
            depth_loss = torch.tensor(0.0).type_as(mfc_loss)
        
        # normal_mask = (targets["normal"].sum(dim=-1) > 0).float() * (targets["mask"] > 0.5).float() * valid_mask.squeeze(-1)
        # normal_gt = torch.nn.functional.normalize(targets["normal"], p=2, dim=-1)
        # normal_pred = torch.nn.functional.normalize(preds["normal"], p=2, dim=-1)
        # normal_l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)
        # normal_cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1))
        # normal_loss = ((normal_l1 + normal_cos) * normal_mask).sum() / (normal_mask.sum() + 1e-10)
        
        loss = color_loss * self.color_weight\
                + eikonal_loss * self.igr_weight \
                + sparse_loss * self.sparse_weight \
                + mfc_loss * self.mfc_weight \
                + smooth_loss * self.smooth_weight \
                + tv_loss * self.tv_weight \
                + pseudo_sdf_loss * self.pseudo_sdf_weight \
                + pseudo_depth_loss * self.pseudo_depth_weight 
        
        loss_outs = {
            "loss": loss,
            "color_loss": color_loss,
            "eikonal_loss": eikonal_loss,
            "sparse_loss": sparse_loss,
            "mfc_loss": mfc_loss,
            "smooth_loss": smooth_loss,
            "tv_loss": tv_loss,
            "depth_loss": depth_loss,
            "pseudo_sdf_loss": pseudo_sdf_loss,
            "pseudo_depth_loss": pseudo_depth_loss,
        }
        
        return loss_outs