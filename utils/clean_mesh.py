import torch
import numpy as np
from skimage import morphology as morph
import torch.nn.functional as F
import trimesh
import open3d as o3d


@torch.no_grad()
def clean_mesh_by_mask(mesh, masks, intrs, c2ws, min_nb_visible=1):
    
    points = torch.from_numpy(mesh.vertices).float().permute(1, 0)  # 3, np
    
    nv, h, w = masks.shape
    
    pts_cam = torch.matmul(c2ws.inverse(), torch.cat([points, torch.ones_like(points[:1])], dim=0)[None])[:, :3]    # nv, 3, np
    pts_img = torch.matmul(intrs[:, :3, :3], pts_cam) # nv, 3, np
    pts_xy = pts_img[:, :2] / torch.clamp(pts_img[:, 2:], 1e-8)
    
    pts_xy[:, 0] = 2 * pts_xy[:, 0] / (w - 1) - 1
    pts_xy[:, 1] = 2 * pts_xy[:, 1] / (h - 1) - 1
    
    in_mask = (pts_xy.abs() <= 1).all(dim=1) & (pts_img[:, -1] > 1e-8)   # nv, np
    
    grid = torch.clamp(pts_xy.permute(0, 2, 1).unsqueeze(1), -10, 10)   # nv, 1, np, 2
    warp_mask = F.grid_sample(masks.unsqueeze(1).float(), grid, align_corners=True).squeeze()   # nv, np
    
    valid_mask = ((warp_mask > 0) * in_mask).sum(dim=0) > min_nb_visible
    
    hull_mask = valid_mask[mesh.faces].all(dim=-1).numpy()

    mesh.update_faces(hull_mask)
    
    return mesh


@torch.no_grad()
def clean_mesh_outside_frustum(mesh, masks, intrs, c2ws, upscale=4):
    
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    
    nv, h, w = masks.shape
    
    all_indices = []
    for i in range(nv):
        intr = intrs[i]
        c2w = c2ws[i]
        mask = masks[i]
        
        ys, xs = torch.meshgrid(torch.linspace(0, h - 1, int(h*upscale)),
                            torch.linspace(0, w - 1, int(w*upscale)))
        p = torch.stack([xs, ys, torch.ones_like(ys)], dim=-1)  # H, W, 3

        # normalized ndc uv coordinates, (-1, 1)
        ndc_u = 2 * xs / (w - 1) - 1
        ndc_v = 2 * ys / (h - 1) - 1
        rays_ndc_uv = torch.stack([ndc_u, ndc_v], dim=-1).view(-1, 2).float()

        p = p.view(-1, 3).float()  # N_rays, 3
        p = torch.matmul(intr.inverse()[None, :3, :3], p[:, :, None]).squeeze()  # N_rays, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # N_rays, 3
        rays_d = torch.matmul(c2w[None, :3, :3], rays_d[:, :, None]).squeeze()  # N_rays, 3
        rays_o = c2w[None, :3, 3].expand(rays_d.shape)  # N_rays, 3
        
        mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), scale_factor=upscale, mode="nearest").squeeze(0).squeeze(0)
        rays_mask = (mask > 0).view(-1)
        
        chunk_size = 5120
        rays_o = rays_o.split(chunk_size)
        rays_d = rays_d.split(chunk_size)
        rays_mask = rays_mask.split(chunk_size)
        
        ind_batch = []
        for rays_o_batch, rays_d_batch, rays_mask_batch in zip(rays_o, rays_d, rays_mask):
            rays_o_batch = rays_o_batch[rays_mask_batch]
            rays_d_batch = rays_d_batch[rays_mask_batch]

            idx_faces_hits = intersector.intersects_first(rays_o_batch.cpu().numpy(), rays_d_batch.cpu().numpy())
            idx_faces_hits = np.unique(idx_faces_hits)
            ind_batch.append(idx_faces_hits)
        ind_batch = np.concatenate(ind_batch, axis=0)
        ind_batch = np.unique(ind_batch)
        all_indices.append(ind_batch)
    
    all_indices = np.concatenate(all_indices, axis=0)
    from collections import Counter
    ind_count = Counter(all_indices)
    num_com_vis = 1
    values = []
    for elem, count in ind_count.items():
        if count >= num_com_vis:
            values.append(elem)
    values.sort()
    
    hull_mask = np.zeros(len(mesh.faces))
    hull_mask[values[1:]] = 1
    mesh.update_faces(hull_mask > 0)
    
    # # clean meshes
    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=500)
    mask = np.zeros(len(mesh.faces), dtype=np.bool)
    mask[np.concatenate(cc)] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    
    return mesh


@torch.no_grad()
def clean_mesh(mesh, masks, intrs, c2ws, dilation_radius=11, min_nb_visible=1, upscale=2):
    
    intrs = intrs.cpu()
    c2ws = c2ws.cpu()
    masks = masks.cpu()
    
    if len(masks.shape) > 3:
        masks = masks.mean(dim=-1)
        
    masks_ = torch.unbind(masks > 0.5)
    dilated_masks = list()
    for m in masks_:
        struct_elem = morph.disk(dilation_radius)
        dilated_masks.append(torch.from_numpy(morph.binary_dilation(m.detach().cpu().numpy(), struct_elem)))
    dilated_masks = torch.stack(dilated_masks)
    
    mesh = clean_mesh_by_mask(mesh, dilated_masks, intrs, c2ws, min_nb_visible)
    
    mesh = clean_mesh_outside_frustum(mesh, masks, intrs, c2ws, upscale=upscale)
    
    return mesh