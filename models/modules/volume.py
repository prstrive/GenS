import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F


class Volume(nn.Module):
    def __init__(self, confs):
        super(Volume, self).__init__()
        
        self.volume_dims = confs.get_list("volume_dims")
    
    def agg_mean_var(self, features, intrs, c2ws, min_vis_view=1):
        """ 
        c2ws: (nv, 4, 4)
        """
        
        volumes = []
        mask_volumes = []
        
        for i in range(len(self.volume_dims)):
            feat_stage = features[i]
            nv, c, height, width = feat_stage.shape
            intrs_stage = intrs.clone()
            intrs_stage[:, :2] *= 0.5**i
            
            with torch.no_grad():
                grid_range = [torch.linspace(-1, 1, self.volume_dims[i]).type_as(intrs_stage) for _ in range(3)]
                # voxel_grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2])[::-1]) # xyz (3, d, h, w)
                voxel_grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2])) # xyz (3, d, h, w)
                world_pts = voxel_grid.reshape(3, -1).unsqueeze(0).repeat(nv, 1, 1)   # nv, 4, npts
                world_pts_homo = torch.cat([world_pts, torch.ones_like(world_pts[:, :1])], dim=1)  
                
                cam_pts = torch.matmul(torch.inverse(c2ws), world_pts_homo)
                img_pts = torch.matmul(intrs_stage, cam_pts)[:, :3]
                xy = img_pts[:, :2] / (img_pts[:, 2:] + 1e-8)
                
                norm_x = xy[:, 0] / ((width - 1) / 2) - 1
                norm_y = xy[:, 1] / ((height - 1) / 2) - 1
                
                grid = torch.stack([norm_x, norm_y], dim=-1)    # nv, npts, 2
                
                mask = (norm_x.abs() <= 1) & (norm_y.abs() <= 1) & (img_pts[:, 2] > 0)  # nv, npts
                mask = mask.unsqueeze(1)    # nv, 1, npts
                
            feat_warp = F.grid_sample(feat_stage, grid.unsqueeze(1), padding_mode='zeros', align_corners=True)  # nv, c, 1, npts
            feat_warp = feat_warp.squeeze(2)    # nv, c, npts
            
            warp_sum = (feat_warp * mask).sum(dim=0)
            warp_sq_sum = ((feat_warp * mask)**2).sum(dim=0)
            mask_sum = mask.sum(dim=0)
            
            inpaint_mask_sum = torch.where(mask_sum<=0, torch.ones_like(mask_sum)*1e-8, mask_sum)
            var = warp_sq_sum / inpaint_mask_sum - (warp_sum / inpaint_mask_sum)**2
            mean = warp_sum / inpaint_mask_sum
            
            volume = torch.cat([mean, var], dim=0).reshape(1, -1, self.volume_dims[i], self.volume_dims[i], self.volume_dims[i])
            mask_volume = (mask_sum > min_vis_view).float().reshape(1, 1, self.volume_dims[i], self.volume_dims[i], self.volume_dims[i])
            
            volumes.append(volume)
            mask_volumes.append(mask_volume)
            
        return volumes, mask_volumes
    
    def agg_adaptive(self, features, intrs, c2ws, min_vis_view=1):
        """ 
        c2ws: (nv, 4, 4)
        """
        
        volumes = []
        mask_volumes = []
        
        for i in range(len(self.volume_dims)):
            feat_stage = features[i]
            nv, c, height, width = feat_stage.shape
            intrs_stage = intrs.clone()
            intrs_stage[:, :2] *= 0.5**i
            
            with torch.no_grad():
                grid_range = [torch.linspace(-1, 1, self.volume_dims[i]).type_as(intrs_stage) for _ in range(3)]
                voxel_grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2])[::-1]) # xyz (3, d, h, w)
                world_pts = voxel_grid.reshape(3, -1).unsqueeze(0).repeat(nv, 1, 1)   # nv, 4, npts
                world_pts_homo = torch.cat([world_pts, torch.ones_like(world_pts[:, :1])], dim=1)  
                
                cam_pts = torch.matmul(torch.inverse(c2ws), world_pts_homo)
                img_pts = torch.matmul(intrs_stage, cam_pts)[:, :3]
                xy = img_pts[:, :2] / (img_pts[:, 2:] + 1e-8)
                
                norm_x = xy[:, 0] / ((width - 1) / 2) - 1
                norm_y = xy[:, 1] / ((height - 1) / 2) - 1
                
                grid = torch.stack([norm_x, norm_y], dim=-1)    # nv, npts, 2
                
                mask = (norm_x.abs() <= 1) & (norm_y.abs() <= 1) & (img_pts[:, 2] > 0)  # nv, npts
                mask = mask.unsqueeze(1)    # nv, 1, npts
                
            feat_warp = F.grid_sample(feat_stage, grid.unsqueeze(1), padding_mode='zeros', align_corners=True)  # nv, c, 1, npts
            feat_warp = feat_warp.squeeze(2)    # nv, c, npts
            
            feats = feat_warp.permute(0, 2, 1).contiguous()
            
            x = self.agg_mlps[i](feats)
            x = x.masked_fill(mask.permute(0, 2, 1).contiguous() == 0, -1e9)
            w = F.softmax(x, dim=0)
            
            volume = (feats * w).sum(dim=0).permute(1, 0).contiguous().reshape(1, -1, self.volume_dims[i], self.volume_dims[i], self.volume_dims[i])
            
            mask_sum = mask.sum(dim=0)
            mask_volume = (mask_sum > min_vis_view).float().reshape(1, 1, self.volume_dims[i], self.volume_dims[i], self.volume_dims[i])
            
            volumes.append(volume)
            mask_volumes.append(mask_volume)
            
        return volumes, mask_volumes