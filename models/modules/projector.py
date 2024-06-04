import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .grid_sample_cuda import cuda_gridsample as cug


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


# - checked for correctness
def grid_sample_3d(volume, optical):
    """
    bilinear sampling cannot guarantee continuous first-order gradient
    mimic pytorch grid_sample function
    The 8 corner points of a volume noted as: 4 points (front view); 4 points (back view)
    fnw (front north west) point
    bse (back south east) point
    :param volume: [B, C, X, Y, Z]
    :param optical: [B, x, y, z, 3], absolute coord, not norm
    :return:
    """
    N, C, ID, IH, IW = volume.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)

    mask_x = (ix > 0) & (ix < IW)
    mask_y = (iy > 0) & (iy < IH)
    mask_z = (iz > 0) & (iz < ID)

    mask_o = mask_x & mask_y & mask_z  # [B, x, y, z]
    mask = mask_o[:, None, :, :, :].repeat(1, C, 1, 1, 1)  # [B, C, x, y, z]

    with torch.no_grad():
        # back north west
        ix_bnw = torch.floor(ix).long()
        iy_bnw = torch.floor(iy).long()
        iz_bnw = torch.floor(iz).long()

        ix_bne = ix_bnw + 1
        iy_bne = iy_bnw
        iz_bne = iz_bnw

        ix_bsw = ix_bnw
        iy_bsw = iy_bnw + 1
        iz_bsw = iz_bnw

        ix_bse = ix_bnw + 1
        iy_bse = iy_bnw + 1
        iz_bse = iz_bnw

        # front view
        ix_fnw = ix_bnw
        iy_fnw = iy_bnw
        iz_fnw = iz_bnw + 1

        ix_fne = ix_bnw + 1
        iy_fne = iy_bnw
        iz_fne = iz_bnw + 1

        ix_fsw = ix_bnw
        iy_fsw = iy_bnw + 1
        iz_fsw = iz_bnw + 1

        ix_fse = ix_bnw + 1
        iy_fse = iy_bnw + 1
        iz_fse = iz_bnw + 1

    # back view
    bnw = (ix_fse - ix) * (iy_fse - iy) * (iz_fse - iz)  # smaller volume, larger weight
    bne = (ix - ix_fsw) * (iy_fsw - iy) * (iz_fsw - iz)
    bsw = (ix_fne - ix) * (iy - iy_fne) * (iz_fne - iz)
    bse = (ix - ix_fnw) * (iy - iy_fnw) * (iz_fnw - iz)

    # front view
    fnw = (ix_bse - ix) * (iy_bse - iy) * (iz - iz_bse)  # smaller volume, larger weight
    fne = (ix - ix_bsw) * (iy_bsw - iy) * (iz - iz_bsw)
    fsw = (ix_bne - ix) * (iy - iy_bne) * (iz - iz_bne)
    fse = (ix - ix_bnw) * (iy - iy_bnw) * (iz - iz_bnw)

    with torch.no_grad():
        # back view
        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

        # front view
        torch.clamp(ix_fnw, 0, IW - 1, out=ix_fnw)
        torch.clamp(iy_fnw, 0, IH - 1, out=iy_fnw)
        torch.clamp(iz_fnw, 0, ID - 1, out=iz_fnw)

        torch.clamp(ix_fne, 0, IW - 1, out=ix_fne)
        torch.clamp(iy_fne, 0, IH - 1, out=iy_fne)
        torch.clamp(iz_fne, 0, ID - 1, out=iz_fne)

        torch.clamp(ix_fsw, 0, IW - 1, out=ix_fsw)
        torch.clamp(iy_fsw, 0, IH - 1, out=iy_fsw)
        torch.clamp(iz_fsw, 0, ID - 1, out=iz_fsw)

        torch.clamp(ix_fse, 0, IW - 1, out=ix_fse)
        torch.clamp(iy_fse, 0, IH - 1, out=iy_fse)
        torch.clamp(iz_fse, 0, ID - 1, out=iz_fse)

    # xxx = volume[:, :, iz_bnw.long(), iy_bnw.long(), ix_bnw.long()]
    volume = volume.view(N, C, ID * IH * IW)
    # yyy = volume[:, :, (iz_bnw * ID + iy_bnw * IW + ix_bnw).long()]

    # back view
    bnw_val = torch.gather(volume, 2,
                           (iz_bnw * ID ** 2 + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(volume, 2,
                           (iz_bne * ID ** 2 + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(volume, 2,
                           (iz_bsw * ID ** 2 + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(volume, 2,
                           (iz_bse * ID ** 2 + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    # front view
    fnw_val = torch.gather(volume, 2,
                           (iz_fnw * ID ** 2 + iy_fnw * IW + ix_fnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fne_val = torch.gather(volume, 2,
                           (iz_fne * ID ** 2 + iy_fne * IW + ix_fne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fsw_val = torch.gather(volume, 2,
                           (iz_fsw * ID ** 2 + iy_fsw * IW + ix_fsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fse_val = torch.gather(volume, 2,
                           (iz_fse * ID ** 2 + iy_fse * IW + ix_fse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (
        # back
            bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
            bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
            bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
            bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W) +
            # front
            fnw_val.view(N, C, D, H, W) * fnw.view(N, 1, D, H, W) +
            fne_val.view(N, C, D, H, W) * fne.view(N, 1, D, H, W) +
            fsw_val.view(N, C, D, H, W) * fsw.view(N, 1, D, H, W) +
            fse_val.view(N, C, D, H, W) * fse.view(N, 1, D, H, W)

    )

    # # * zero padding
    # out_val = torch.where(mask, out_val, torch.zeros_like(out_val).float().to(out_val.device))

    return out_val


def lookup_volume(pts, volume, sample_mode="grad"):
    """
    pts: (n_pts, 3), default range from -1 to 1
    """
    pts = pts.reshape(-1, 3)
    n_pts, _ = pts.shape
    x = pts.unsqueeze(0).unsqueeze(0).unsqueeze(0).flip(dims=[-1])
    # x = pts.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    if isinstance(volume, torch.Tensor):
        if sample_mode == "grad":
            # feat = grid_sample_3d(volume, x)
            feat = cug.grid_sample_3d(volume, x, padding_mode='zeros', align_corners=True)
        else:
            feat = F.grid_sample(volume, x, mode=sample_mode)
        feat = feat.reshape(-1, n_pts).permute(1, 0).contiguous()
    else:
        feats = []
        for volume_feat in volume:
            if sample_mode == "grad":
                # feat = grid_sample_3d(volume_feat, x)
                feat = cug.grid_sample_3d(volume_feat, x, padding_mode='zeros', align_corners=True)
            else:
                feat = F.grid_sample(volume_feat, x, mode=sample_mode)
            feat = feat.reshape(-1, n_pts).permute(1, 0).contiguous()
            feats.append(feat)
        feat = torch.cat(feats, dim=-1)

    return feat


def equirect2sphere(pts):
    """ 
    pts: (n_pts, 3), x y z
    """
    x, y, z = torch.split(pts, 1, dim=1)
    # dis_to_center = torch.linalg.norm(pts, ord=2, dim=1, keepdim=True) + 1e-5
    dis_to_center = torch.linalg.norm(pts, ord=2, dim=1, keepdim=True).clip(1.0, 1e10)
    
    x_, y_, z_ = x / dis_to_center, y / dis_to_center, z / dis_to_center
    
    r = 1 / dis_to_center
    theta = torch.asin(z_)
    phi = torch.atan2(y_, x_)
    
    sphe_pts = torch.cat([theta, phi, r], dim=1)
    
    return sphe_pts

def lookup_sphe_volume(sphe_pts, volume, sample_mode="grad"):
    theta, phi, r = torch.split(sphe_pts, 1, dim=1)
    theta = theta / (np.pi / 2)  # to [-1, 1]
    phi = phi / np.pi   # to [-1, 1], min -pi, max pi
    r = ((r - 1e-10) / (1 - 1e-10) - 0.5) * 2   # to [-1, 1], min 1e-10, max 1
    
    norm_sphe_pts = torch.cat([theta, phi, r], dim=1)
    feat = lookup_volume(norm_sphe_pts, volume, sample_mode)
    
    return feat


def compute_angle(pts, ref_c2w, src_c2ws):
    n_srcs = src_c2ws.shape[0]
    ref_c2w = ref_c2w.unsqueeze(0).repeat(n_srcs, 1, 1)
    ray2ref_pose = (ref_c2w[:, :3, 3].unsqueeze(1) - pts.unsqueeze(0))
    ray2ref_pose /= (torch.norm(ray2ref_pose, dim=-1, keepdim=True) + 1e-6)
    ray2src_pose = (src_c2ws[:, :3, 3].unsqueeze(1) - pts.unsqueeze(0))
    ray2src_pose /= (torch.norm(ray2src_pose, dim=-1, keepdim=True) + 1e-6)
    ray_diff = ray2ref_pose - ray2src_pose
    ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
    ray_diff_dot = torch.sum(ray2ref_pose * ray2src_pose, dim=-1, keepdim=True)
    ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
    ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
    ray_diff = ray_diff.permute(1, 0, 2).contiguous() # (n_pts, n_srcs, 4)
    return ray_diff


def lookup_feature(pts, imgs, intrs, c2ws, features):
    """
    pts: (n_rays*n_samples, 3)
    intrs: (n_srcs, 4, 4)
    c2ws: (n_srcs, 4, 4)
    """

    intrs, c2ws, ref_c2w = intrs[1:].clone(), c2ws[1:].clone(), c2ws[0].clone()

    ray_diff = compute_angle(pts, ref_c2w, c2ws)

    n_srcs = intrs.shape[0]

    n_pts = pts.shape[0]
    pts_ = pts.permute(1, 0).contiguous().reshape(3, -1)

    if not isinstance(features, list):
        features = [features]

    warped_f_cas = []
    mask_cas = []
    for i, feat in enumerate(features):
        
        intrs_ = intrs.clone()
        intrs_[:, :2] = intrs_[:, :2] * (0.5 ** i)
        with torch.no_grad():
            _, c, height, width = feat.shape       
            
            cam_srcs = torch.matmul(torch.inverse(c2ws), torch.cat([pts_, torch.ones_like(pts_[:1])], dim=0)[None, :, :])[:, :3]
            cam_srcs = torch.matmul(intrs_[:, :3, :3], cam_srcs)
            xy_srcs = cam_srcs[:, :2] / cam_srcs[:, 2:] # [n_srcs, 2, n_rays*n_samples]

            norm_x = xy_srcs[:, 0] / ((width - 1) / 2) - 1
            norm_Y = xy_srcs[:, 1] / ((height - 1) / 2) - 1

            mask = (cam_srcs[:, 2] > 0) & (xy_srcs[:, 0] >= 0) & (xy_srcs[:, 0] < width) & (xy_srcs[:, 1] >= 0) & (xy_srcs[:, 1] < height)
            mask = mask.reshape(n_srcs, n_pts).permute(1, 0).contiguous()   # (n_rays*n_samples, n_srcs)
            mask_cas.append(mask)

            grid = torch.stack([norm_x, norm_Y], dim=-1).unsqueeze(dim=2)    # [n_srcs, n_rays*n_samples, 1, 2]

        src_feats = feat[1:]

        warped_f = F.grid_sample(src_feats, grid, mode='bilinear', padding_mode='zeros').type(torch.float32)
        warped_f = warped_f.reshape(n_srcs, c, n_pts).permute(2, 0, 1).contiguous()   # (n_rays*n_samples, n_srcs, c)
        warped_f_cas.append(warped_f)

        if i == 0:
            src_imgs = imgs[1:]
            warped_rgb = F.grid_sample(src_imgs, grid, mode='bilinear', padding_mode='zeros').type(torch.float32)
            warped_rgb = warped_rgb.reshape(n_srcs, 3, n_pts).permute(2, 0, 1).contiguous()   # (n_rays*n_samples, n_srcs, 3)
    
    warped_f_cas = torch.cat(warped_f_cas, dim=2)
    mask_cas = torch.stack(mask_cas, dim=-1).all(dim=-1)

    return torch.cat([warped_rgb, warped_f_cas], dim=2), ray_diff, mask_cas


# have bug, (b, h, w, c) 2 (b, c, h, w)
def surface_patch_warp(pts_sdf0, gradients_sdf0, images, intrinsics, poses, patch_size=11):
    """ 
    pts: (n_pts, 3)
    normal: (n_pts, 3)
    intrs: (n_view, 4, 4)
    c2ws: (n_view, 4, 4)
    images: (n_view, c, h, w)
    """
    
    batch_size = pts_sdf0.shape[0]
    
    intrinsics_inv = torch.inverse(intrinsics)
    
    project_xyz = torch.matmul(poses[0, :3, :3].permute(1, 0).contiguous(), pts_sdf0.permute(0, 2, 1).contiguous())
    t = - torch.matmul(poses[0, :3, :3].permute(1, 0).contiguous(), poses[0, :3, 3, None])
    project_xyz = project_xyz + t
    pts_sdf0_ref = project_xyz
    project_xyz = torch.matmul(intrinsics[0, :3, :3], project_xyz)  # [batch_size, 3, 1]
    disp_sdf0 = torch.matmul(gradients_sdf0, pts_sdf0_ref)  # [batch_size, 1, 1]

    # Compute Homography
    K_ref_inv = intrinsics_inv[0, :3, :3]
    K_src = intrinsics[1:, :3, :3]
    num_src = K_src.shape[0]
    R_ref_inv = poses[0, :3, :3]
    R_src = poses[1:, :3, :3].permute(0, 2, 1).contiguous()
    C_ref = poses[0, :3, 3]
    C_src = poses[1:, :3, 3]
    R_relative = torch.matmul(R_src, R_ref_inv)
    C_relative = C_ref[None, ...] - C_src
    tmp = torch.matmul(R_src, C_relative[..., None])
    tmp = torch.matmul(tmp[None, ...].expand(batch_size, num_src, 3, 1), gradients_sdf0.expand(batch_size, num_src, 3)[..., None].permute(0, 1, 3, 2))  # [Batch_size, num_src, 3, 1]
    tmp = R_relative[None, ...].expand(batch_size, num_src, 3, 3) + tmp / (disp_sdf0[..., None] + 1e-10)
    tmp = torch.matmul(K_src[None, ...].expand(batch_size, num_src, 3, 3), tmp)
    Hom = torch.matmul(tmp, K_ref_inv[None, None, ...])

    pixels_x = project_xyz[:, 0, 0] / (project_xyz[:, 2, 0] + 1e-8)
    pixels_y = project_xyz[:, 1, 0] / (project_xyz[:, 2, 0] + 1e-8)
    pixels = torch.stack([pixels_x, pixels_y], dim=-1).float()
    h_patch_size = patch_size // 2
    total_size = (h_patch_size * 2 + 1) ** 2
    offsets = torch.arange(-h_patch_size, h_patch_size + 1)
    offsets = torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2).type_as(pixels)    # [1, patch*patch, 2]
    pixels_patch = pixels.view(batch_size, 1, 2) + offsets.float()  # [batch_size, 121, 2]

    ref_image = images[0, ...]
    src_images = images[1:, ...]
    h, w = ref_image.shape[-2:]
    
    grid = patch_homography(Hom, pixels_patch)
    grid[:, :, 0] = 2 * grid[:, :, 0] / (w - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (h - 1) - 1.0
    if len(src_images.shape)==3:
        sampled_gray_val = F.grid_sample(src_images.unsqueeze(1), grid.view(num_src, -1, 1, 2), align_corners=True)
    elif len(src_images.shape)==4:
        sampled_gray_val = F.grid_sample(src_images, grid.view(num_src, -1, 1, 2), align_corners=True)
    sampled_gray_val = sampled_gray_val.view(num_src, -1, batch_size, total_size).permute(0, 2, 3, 1).contiguous()  # [nsrc, batch_size, 121, c]
    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (w - 1) - 1.0
    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (h - 1) - 1.0
    grid = pixels_patch.detach()
    if len(ref_image.shape)==2:
        ref_gray_val = F.grid_sample(ref_image[None, None, ...], grid.view(1, -1, 1, 2), align_corners=True)
    elif len(ref_image.shape)==3:
        ref_gray_val = F.grid_sample(ref_image[None, ...], grid.view(1, -1, 1, 2), align_corners=True)
    ref_gray_val = ref_gray_val.view(1, -1, batch_size, total_size).permute(0, 2, 3, 1).contiguous()
    
    return ref_gray_val, sampled_gray_val

    
def patch_homography(H, uv):
    # H: [batch_size, nsrc, 3, 3]
    # uv: [batch_size, 121, 2]
    N, Npx = uv.shape[:2]
    H = H.permute(1, 0, 2, 3).contiguous()
    Nsrc = H.shape[0]
    H = H.view(Nsrc, N, -1, 3, 3)
    ones = torch.ones(uv.shape[:-1], device=uv.device).unsqueeze(-1)
    hom_uv = torch.cat((uv, ones), dim=-1)

    tmp = torch.einsum("vprik,pok->vproi", H, hom_uv)
    tmp = tmp.reshape(Nsrc, -1, 3)

    grid = tmp[..., :2] / (tmp[..., 2:] + 1e-8)

    return grid

def patch_homography2(H, uv):
    # H: [batch_size, nsrc, 3, 3]
    # uv: [nsrc, batch_size, 121, 2]
    Nsrc, N, Npx = uv.shape[:3]
    H = H.permute(1, 0, 2, 3).contiguous()
    H = H.view(Nsrc, N, -1, 3, 3)
    ones = torch.ones(uv.shape[:-1], device=uv.device).unsqueeze(-1)
    hom_uv = torch.cat((uv, ones), dim=-1)  # [nsrc, batch_size, 121, 3]

    tmp = torch.einsum("vprik,vpok->vproi", H, hom_uv)
    tmp = tmp.reshape(Nsrc, -1, 3)

    grid = tmp[..., :2] / (tmp[..., 2:] + 1e-8)

    return grid