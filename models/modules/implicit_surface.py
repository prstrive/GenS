import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import mcubes

from .sdf_network import  SDFNetwork
from .blending_network import BlendingNetwork
from .rendering_network import RenderingNetwork
from .variance_network import SingleVarianceNetwork
from .projector import lookup_feature, lookup_volume, surface_patch_warp


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).type_as(weights)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).type_as(weights)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class ImplicitSurface(nn.Module):
    def __init__(self, confs):
        super(ImplicitSurface, self).__init__()
        
        self.finetune_mode = confs.get_int("finetune_mode", default=0)
        self.n_samples = confs.get_int("render.n_samples")
        self.n_importance = confs.get_int("render.n_importance")
        self.up_sample_steps = confs.get_int("render.up_sample_steps")
        self.perturb = confs.get_float("render.perturb")
        
        self.sdf_network = SDFNetwork(**confs["sdf_network"])
        if self.finetune_mode:
            # for finetuning
            self.color_network = RenderingNetwork(**confs["color_network"])
        else:
            self.color_network = BlendingNetwork(**confs["color_network"])
        self.deviation_network = SingleVarianceNetwork(**confs["variance_network"])
        
    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, mask_volumes, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3

        voxel_mask = lookup_volume(pts.reshape(-1, 3), mask_volumes, sample_mode='nearest').any(dim=-1, keepdim=True)
        voxel_mask = voxel_mask.reshape(batch_size, n_samples)
        voxel_mask = voxel_mask[:, :-1] * voxel_mask[:, 1:]

        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = ((radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)) & (voxel_mask > 0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]).type_as(cos_val), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).type_as(alpha), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, volumes, mask_volumes, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            pts_mask = lookup_volume(pts, mask_volumes, sample_mode='nearest').any(dim=-1, keepdim=True)
            pts_mask_bool = (pts_mask > 0).view(-1)
            if torch.sum(pts_mask_bool.float()) < 1:
                pts_mask_bool[:10] = True
            new_sdf = torch.ones_like(pts[:, :1]) * 100
            new_sdf[pts_mask_bool] = self.sdf_network.sdf(pts[pts_mask_bool], volumes)
            new_sdf = new_sdf.reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1).type_as(index)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
    
    def tv_regularization(self, volume_feat_cas, volume_mask_cas=None):
        if volume_mask_cas is None:
            volume_mask_cas = [torch.ones_like(vol[:, :1]) for vol in volume_feat_cas]
        
        total_tv = 0
        for i, (volume_fea, volume_mask) in enumerate(zip(volume_feat_cas, volume_mask_cas)):
            mx = (volume_mask[:, :, 1:, :, :] * volume_mask[:, :, :-1, :, :]) > 0
            my = (volume_mask[:, :, :, 1:, :] * volume_mask[:, :, :, :-1, :]) > 0
            mz = (volume_mask[:, :, :, :, 1:] * volume_mask[:, :, :, :, :-1]) > 0
            
            tx = ((volume_fea[:, :, 1:, :, :] - volume_fea[:, :, :-1, :, :])**2 * mx).sum() / (mx.sum() + 1e-8)
            ty = ((volume_fea[:, :, :, 1:, :] - volume_fea[:, :, :, :-1, :])**2 * my).sum() / (mx.sum() + 1e-8)
            tz = ((volume_fea[:, :, :, :, 1:] - volume_fea[:, :, :, :, :-1])**2 * mz).sum() / (mx.sum() + 1e-8)
            tv = torch.sqrt(tx + ty + tz)
            total_tv = total_tv + tv * 0.5 ** i
        return total_tv
    
    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    volumes, mask_volumes, features, match_features, imgs, intrs, c2ws,
                    cos_anneal_ratio, step):

        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).type_as(dists)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        voxel_mask = lookup_volume(pts, mask_volumes, sample_mode='nearest').any(dim=-1, keepdim=True)
        pts_mask_bool = (voxel_mask > 0).view(-1)
        if torch.sum(pts_mask_bool.float()) < 1:
            pts_mask_bool[:10] = True
                
        sdf_nn_output = self.sdf_network(pts[pts_mask_bool], volumes)

        sdf = torch.ones_like(pts[:, :1]) * 100
        sdf[pts_mask_bool] = sdf_nn_output[:, :1]
        feature_vector_valid = sdf_nn_output[:, 1:]
        feature_vector = torch.zeros(pts.shape[0], feature_vector_valid.shape[1]).type_as(feature_vector_valid)
        feature_vector[pts_mask_bool] = feature_vector_valid
        
        gradients = torch.zeros_like(pts)
        smooth = torch.zeros_like(pts)
        gradients_valid, smooth_valid = self.sdf_network.gradient(pts.clone()[pts_mask_bool], volumes)
        gradients[pts_mask_bool] = gradients_valid
        smooth[pts_mask_bool] = smooth_valid
        
        if self.finetune_mode:
            # rendering color net for finetuning
            sampled_color = torch.zeros_like(pts)
            sampled_color_valid = self.color_network(pts[pts_mask_bool], gradients[pts_mask_bool], dirs[pts_mask_bool], feature_vector[pts_mask_bool])
            sampled_color[pts_mask_bool] = sampled_color_valid
            sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
            valid_mask = torch.ones(batch_size, 1).type_as(pts)
        else:
            # blending color net for training
            sampled_color = torch.zeros_like(pts)
            mask = torch.zeros(pts.shape[0], imgs.shape[0]-1).bool().to(pts.device)
            feat_views, ray_diff, mask_valid = lookup_feature(pts[pts_mask_bool], imgs, intrs, c2ws, features)
            # mask = (mask.float() * (torch.abs(pts)<1).all(dim=-1, keepdim=True).float()) > 0
            sampled_color_valid = self.color_network(feat_views, ray_diff, mask_valid)
            sampled_color[pts_mask_bool] = sampled_color_valid
            sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
            mask[pts_mask_bool] = mask_valid
            minimum_vas_view = 1 #max(2, int(mask.shape[-1]*0.4))
            valid_mask = mask.reshape(batch_size, n_samples, -1).detach().float()
            valid_mask = (valid_mask.sum(dim=2) > minimum_vas_view).float().sum(dim=1, keepdim=True) > 8 # (n_ray, 1)

        inv_s = self.deviation_network(torch.zeros([1, 3]).type_as(rays_o))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        iter_cos = iter_cos * voxel_mask.reshape(-1, 1)

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        alpha = alpha * voxel_mask.reshape(batch_size, n_samples)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach() * voxel_mask.reshape(batch_size, n_samples)
        relax_inside_sphere = (pts_norm < 1.2).float().detach() * voxel_mask.reshape(batch_size, n_samples)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).type_as(alpha), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        weights_sum_fg = weights[:, :n_samples].sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
    
        normal = (gradients.reshape(batch_size, n_samples, 3) * weights[:, :, None]).sum(dim=1) # (n_rays, 3)
        rot = torch.inverse(c2ws[0, :3, :3])
        normal = torch.matmul(rot[None, :, :], normal[:, :, None]).squeeze(-1)
        
        cam_rays_d = torch.matmul(torch.inverse(c2ws[0, None, :3, :3]), rays_d[:, :, None]).squeeze()
        render_depth = (mid_z_vals * weights).sum(dim=1) * cam_rays_d[:, 2] # (nr, ) z_val * cos = depth

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        smooth_sur_point = (smooth.reshape(batch_size, n_samples, 3) * weights[:, :n_samples, None].detach() * inside_sphere[:, :, None]).sum(dim=1)
        smooth_error = torch.linalg.norm(smooth_sur_point, ord=2, dim=-1).abs().mean()
        # smooth_error = (torch.linalg.norm(smooth, ord=2, dim=-1).reshape(batch_size, n_samples) * inside_sphere).sum() / (inside_sphere.sum() + 1e-5)

        pts_random = torch.rand([1024, 3]).type_as(rays_o) * 2 - 1  # normalized to (-1, 1)
        sdf_random = self.sdf_network.sdf(pts_random, volumes)
        
        # tv reg
        tv_reg = self.tv_regularization(volumes, mask_volumes)
        
        sdf_d = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
        voxel_mask_d = voxel_mask.reshape(batch_size, n_samples)
        pre_mask, next_mask = voxel_mask_d[:, :-1], voxel_mask_d[:, 1:]
        valid_mask_d = ((pre_mask * next_mask) > 0).float()
        
        sign = prev_sdf * next_sdf
        sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
        idx = reversed(torch.Tensor(range(1, n_samples)).cuda())
        tmp = torch.einsum("ab,b->ab", (sign, idx))
        tmp = tmp * valid_mask_d
        
        prev_idx = torch.argmax(tmp, 1, keepdim=True)
        next_idx = prev_idx + 1
        prev_inside_sphere = torch.gather(inside_sphere, 1, prev_idx)
        next_inside_sphere = torch.gather(inside_sphere, 1, next_idx)
        mid_inside_sphere = (0.5 * (prev_inside_sphere + next_inside_sphere) > 0.5).float()
        mid_inside_sphere = mid_inside_sphere * (tmp.sum(dim=1, keepdim=True)>0).float()
        
        grad_d = gradients.reshape(batch_size, n_samples, 3).detach()
        grad1 = torch.gather(grad_d, 1, prev_idx.unsqueeze(-1).repeat(1, 1, 3))
        grad2 = torch.gather(grad_d, 1, next_idx.unsqueeze(-1).repeat(1, 1, 3))
        cos_d = (grad1 * grad2).sum(dim=-1) / (torch.linalg.norm(grad1, ord=2, dim=-1, keepdim=False) * torch.linalg.norm(grad2, ord=2, dim=-1, keepdim=False) + 1e-8)
        # print("cos:", torch.min(cos_d).item(), torch.max(cos_d).item(), torch.median(cos_d).item())
        mid_inside_sphere = mid_inside_sphere * (cos_d > 0.5)
        
        sdf1 = torch.gather(sdf_d, 1, prev_idx)
        sdf2 = torch.gather(sdf_d, 1, next_idx)
        z_vals1 = torch.gather(mid_z_vals, 1, prev_idx)
        z_vals2 = torch.gather(mid_z_vals, 1, next_idx)
        z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
        
        # d_pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_sdf0[..., :, None]  # [batch_size, 1, 3]
        # cam_pts = torch.matmul(torch.inverse(c2ws)[0, None], torch.cat([d_pts, torch.ones_like(d_pts[..., :1])], dim=-1).squeeze(1)[..., None]).squeeze()[:, :3]
        # sdf_depth = cam_pts[:, -1:] * mid_inside_sphere  # (batch, 1)
        
        sdf_depth = z_vals_sdf0 * cam_rays_d[:, 2:3] # (nr, 1) z_val * cos = depth
        sdf_depth = sdf_depth * mid_inside_sphere
        
        z_vals_sdf0 = torch.where(z_vals_sdf0 < 0, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        max_z_val = torch.max(z_vals)
        z_vals_sdf0 = torch.where(z_vals_sdf0 > max_z_val, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        pts_sdf0 = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_sdf0[..., :, None]  # [batch_size, 1, 3]
        gradients_sdf0, smooth_sdf0 = self.sdf_network.gradient(pts_sdf0.reshape(-1, 3), volumes)
        
        gradients_sdf0 = gradients_sdf0.reshape(batch_size, 1, 3)
        gradients_sdf0_norm = torch.linalg.norm(gradients_sdf0, ord=2, dim=-1, keepdim=True)
        gradients_sdf0_norm = torch.where(gradients_sdf0_norm<=0, torch.ones_like(gradients_sdf0_norm)*1e-8, gradients_sdf0_norm)
        gradients_sdf0 = gradients_sdf0 / gradients_sdf0_norm
        gradients_sdf0 = torch.matmul(c2ws[0, :3, :3].permute(1, 0).contiguous()[None, ...], gradients_sdf0.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous().detach()
        
        if step is None or step < 5:
            warp_feats0 = features[0].detach()
            warp_feats1 = features[1].detach()
            warp_feats1 =  F.interpolate(warp_feats1, size=warp_feats0.shape[-2:], mode="bilinear")
            warp_feats2 = features[2].detach()
            warp_feats2 =  F.interpolate(warp_feats2, size=warp_feats0.shape[-2:], mode="bilinear")
            warp_feats = torch.cat([warp_feats0, warp_feats1, warp_feats2], dim=1).detach()
        else:
            warp_feats0 = match_features[0].detach()
            warp_feats1 = match_features[1].detach()
            warp_feats1 =  F.interpolate(warp_feats1, size=warp_feats0.shape[-2:], mode="bilinear")
            warp_feats2 = match_features[2].detach()
            warp_feats2 =  F.interpolate(warp_feats2, size=warp_feats0.shape[-2:], mode="bilinear")
            warp_feats = torch.cat([warp_feats0, warp_feats1, warp_feats2], dim=1).detach()
        
        ref_gray_val, sampled_gray_val = surface_patch_warp(pts_sdf0, gradients_sdf0, warp_feats, intrs, c2ws)

        return {
            'ref_gray_val': ref_gray_val,
            'sampled_gray_val': sampled_gray_val,
            'mid_inside_sphere': mid_inside_sphere,
            'smooth_error': smooth_error,
            'tv_reg': tv_reg,
            'color_fine': color,
            'render_depth': render_depth,
            'valid_mask': valid_mask,
            'sparse_sdf': torch.cat([sdf_random, sdf]),
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            'normal': normal,
            's_val': 1.0 / inv_s,
            'weights': weights,
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'sdf_depth': sdf_depth
        }
        
    def render(self, rays_o, rays_d, near, far, volumes, mask_volumes, imgs, features, match_features, intrs, c2ws, cos_anneal_ratio, step):
        batch_size = len(rays_o)
        near = near.repeat(batch_size, 1)
        far = far.repeat(batch_size, 1)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).type_as(near)
        z_vals = near + (far - near) * z_vals[None, :]

        perturb = self.perturb

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).type_as(z_vals)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                pts = pts.reshape(-1, 3)
                pts_mask = lookup_volume(pts, mask_volumes, sample_mode='nearest').any(dim=-1, keepdim=True)
                pts_mask_bool = (pts_mask > 0).view(-1)
                if torch.sum(pts_mask_bool.float()) < 1:
                    pts_mask_bool[:10] = True
                sdf = torch.ones_like(pts[:, :1]) * 100
                sdf[pts_mask_bool] = self.sdf_network.sdf(pts[pts_mask_bool], volumes)
                sdf = sdf.reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                mask_volumes,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  volumes,
                                                  mask_volumes,
                                                  last=(i + 1 == self.up_sample_steps))

        
        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    volumes, mask_volumes, features, match_features, imgs, intrs, c2ws,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    step=step)

        return ret_fine
    
    def extract_geometry(self, volumes, bound_min, bound_max, resolution, threshold):
        N = 64
        X = torch.linspace(bound_min[0], bound_max[0], resolution).type_as(bound_min).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).type_as(bound_min).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).type_as(bound_min).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # (n_points, 3)
                        val = -self.sdf_network.sdf(pts, volumes).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()
        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        return vertices, triangles
    
    def validate(self, rays_o, rays_d, near, far, volumes, mask_volumes, imgs, features, match_features, intrs, c2ws, bound_min, bound_max, hw, cos_anneal_ratio=1.0, step=None, extract_geometry=True, mesh_resolution=512, threshold=0.0):
        outputs = {}
        if extract_geometry:
            vertices, triangles = self.extract_geometry(volumes, bound_min, bound_max, mesh_resolution, threshold)
            outputs["vertices"] = vertices
            outputs["triangles"] = triangles
        
        height, width = hw
        rays_os = rays_o.split(256)
        rays_ds = rays_d.split(256)

        out_rgb_fine = []
        out_normal_fine = []
        out_sdf_depth = []
        out_render_depth = []
        for rays_o, rays_d in zip(rays_os, rays_ds):
            render_outs = self.render(rays_o, rays_d, near, far, volumes, mask_volumes, imgs, features, match_features, intrs, c2ws, cos_anneal_ratio, step)
            out_rgb_fine.append(render_outs['color_fine'].detach().cpu())
            # n_samples = self.n_samples + self.n_importance
            normals = render_outs['gradients'] * render_outs['weights'][:, :render_outs['gradients'].shape[1], None]
            normals = normals * render_outs['inside_sphere'][..., None]
            normals = normals.sum(dim=1).detach().cpu().numpy()
            out_normal_fine.append(normals)
            out_sdf_depth.append(render_outs['sdf_depth'].detach().cpu().numpy())
            out_render_depth.append(render_outs['render_depth'].detach().cpu().numpy())
            
        color_fine = torch.cat(out_rgb_fine, dim=0)
        img_fine = (color_fine.numpy().reshape([height, width, 3]) * 256).clip(0, 255)
        normal_img = np.concatenate(out_normal_fine, axis=0)
        rot = np.linalg.inv(c2ws[0, :3, :3].detach().cpu().numpy())
        normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                        .reshape([height, width, 3]) * 128 + 128).clip(0, 255)
        out_sdf_depth = np.concatenate(out_sdf_depth, axis=0).reshape([height, width])
        out_render_depth = np.concatenate(out_render_depth, axis=0).reshape([height, width])

        outputs["color_fine"] = color_fine
        outputs["img_fine"] = img_fine
        outputs["normal_img"] = normal_img
        outputs["sdf_depth"] = out_sdf_depth
        outputs["render_depth"] = out_render_depth
        
        return outputs
        
    def forward(self, mode, ipts, volumes, mask_volumes, features, match_features, cos_anneal_ratio=1.0, step=None):
        imgs = ipts["imgs"]  # (nv, 3, h, w)
        intrs = ipts["intrs"]   # (nv, 4, 4)
        c2ws = ipts["c2ws"] # (nv, 4, 4)
        rays_o = ipts["rays_o"] # (nr, 3)
        rays_d = ipts["rays_d"] # (nr, 3)
        near = ipts["near"] # (nr, 1)
        far = ipts["far"]   # (nr, 1)
        
        if mode == "val":
            bound_min_batch = ipts["bound_min"]
            bound_max_batch = ipts["bound_max"]
            hw_batch = ipts["hw"]
            outputs = self.validate(rays_o, rays_d, near, far, volumes, mask_volumes, imgs, features, match_features, intrs, c2ws, bound_min_batch, bound_max_batch, hw_batch, cos_anneal_ratio, step)
        else:
            outputs = self.render(rays_o, rays_d, near, far, volumes, mask_volumes, imgs, features, match_features, intrs, c2ws, cos_anneal_ratio, step)
        
        if "pseudo_pts" in ipts:
            pseudo_pts = ipts["pseudo_pts"]
            pts_mask = lookup_volume(pseudo_pts, mask_volumes, sample_mode='nearest').any(dim=-1, keepdim=True)
            pts_mask_bool = (pts_mask > 0).view(-1)
            pseudo_sdf = torch.zeros_like(pseudo_pts[:, :1])
            if torch.sum(pts_mask_bool.float()) < 1:
                raise("No valid pseudo pts!")
            pseudo_sdf[pts_mask_bool] = self.sdf_network.sdf(pseudo_pts[pts_mask_bool], volumes)
            outputs["pseudo_sdf"] = pseudo_sdf
        
        return outputs