import re
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # ? why need transpose here
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # ! return cam2world matrix here


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


class BMVSDatasetFinetune(Dataset):
    def __init__(self, confs, mode):
        super(BMVSDatasetFinetune, self).__init__()

        self.mode = mode
        self.data_dir = confs['data_dir']
        self.interval_scale = confs.get_float('interval_scale')
        self.num_interval = confs.get_int('num_interval')
        self.img_hw = confs['img_hw']
        self.n_rays = confs.get_int('n_rays')
        self.factor = confs.get_float('factor')
        self.num_views = confs.get_int('num_views')

        self.scene = confs.get_string('scene')
        self.ref_view = confs.get_int('ref_view')
        self.val_res_level = confs.get_int('val_res_level', default=1)
        
        self.pairs = self.get_pairs()
        self.all_views = [self.ref_view] + list(self.pairs[self.ref_view])[:(self.num_views-1)]
        
        self.intrs, self.c2ws, self.near_fars, self.scale_factor, self.trans_mat, self.scale_mat = self.read_cam_info()
        self.intrs = torch.from_numpy(np.stack(self.intrs).astype(np.float32))
        self.c2ws = torch.from_numpy(np.stack(self.c2ws).astype(np.float32))
        self.near_fars = torch.from_numpy(np.stack(self.near_fars).astype(np.float32))
        self.scale_mat = torch.from_numpy(self.trans_mat @ self.scale_mat)
        
        self.images_lis = [os.path.join(self.data_dir, self.scene, 'blended_images/{:0>8}.jpg'.format(vid)) for vid in self.all_views]
        self.masks_lis = [os.path.join(self.data_dir, self.scene, 'blended_images/{:0>8}_masked.jpg'.format(vid)) for vid in self.all_views]
        self.images = [np.array(Image.open(im_name), dtype=np.float32) / 256.0 for im_name in self.images_lis]
        self.images = np.stack([cv2.resize(img, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST) for img in self.images])
        self.images = torch.from_numpy(self.images.astype(np.float32))
        self.masks = [np.array(Image.open(im_name), dtype=np.float32) for im_name in self.masks_lis]
        self.masks = np.stack([(np.mean(cv2.resize(mask, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST), axis=-1)>0).astype(np.float32) for mask in self.masks])
        self.masks = torch.from_numpy(self.masks.astype(np.float32))

    def get_pairs(self):
        
        pair_file = f"{self.scene}/cams/pair.txt"
        print("Using existing pair file...")
        with open(os.path.join(self.data_dir, pair_file)) as f:
            num_viewpoint = int(f.readline())
            pairs = [[]] * num_viewpoint
            # viewpoints (49)
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                pairs[ref_view] = np.array(src_views[:10])
        pairs = np.array(pairs)

        return pairs

    def read_cam_info(self):
        intrs = []
        w2cs = []
        near_fars = []
        for vid in self.all_views:
            filename = os.path.join(self.data_dir, self.scene, 'cams/{:0>8}_cam.txt').format(vid)

            with open(filename) as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
            # extrinsics: line [1,5), 4x4 matrix
            extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
            # intrinsics: line [7-10), 3x3 matrix
            intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
            # intrinsics[:2] *= 4
            intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
            intrinsics_[:3, :3] = intrinsics
            # depth_min & depth_interval: line 11
            depth_min = float(lines[11].split()[0])
            depth_interval = float(lines[11].split()[1]) * self.interval_scale
            depth_max = depth_min + depth_interval * self.num_interval

            intrinsics_[0] *= self.img_hw[1] / 768
            intrinsics_[1] *= self.img_hw[0] / 576

            intrs.append(intrinsics_)
            w2cs.append(extrinsics)
            near_fars.append([depth_min, depth_max])
            
        w2c_ref = w2cs[0]
        w2c_ref_inv = np.linalg.inv(w2c_ref)
        new_w2cs = []
        for w2c in w2cs:
            new_w2cs.append(w2c @ w2c_ref_inv)
                        
        scale_mat, scale_factor = self.get_scale_mat(self.img_hw, intrs, new_w2cs, near_fars, factor=self.factor)
        
        c2ws = []
        new_near_fars = []
        new_intrs = []
        for intr, w2c in zip(intrs, new_w2cs):
            P = intr @ w2c @ scale_mat
            P = P[:3, :4]
            new_intr, c2w = load_K_Rt_from_P(None, P)
            c2ws.append(c2w)
            new_intrs.append(new_intr)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2)).astype(np.float32)
            near = dist - 1
            far = dist + 1
            new_near_fars.append([0.95 * near, 1.05 * far])
        
        return new_intrs, c2ws, new_near_fars, scale_factor, w2c_ref_inv, scale_mat

    def get_scale_mat(self, img_hw, intrs, w2cs, near_fars, factor=0.8):
        bnds = np.zeros((3, 2))
        bnds[:, 0] = np.inf
        bnds[:, 1] = -np.inf
        im_h, im_w = img_hw

        for intr, w2c, near_far in zip(intrs, w2cs, near_fars):
            min_depth, max_depth = near_far

            view_frust_pts = np.stack([
                (np.array([0, 0, im_w, im_w, 0, 0, im_w, im_w]) - intr[0, 2]) * np.array(
                    [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]) / intr[0, 0],
                (np.array([0, im_h, 0, im_h, 0, im_h, 0, im_h]) - intr[1, 2]) * np.array(
                    [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]) / intr[1, 1],
                np.array([min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth])
            ])
            view_frust_pts = view_frust_pts.astype(np.float32)
            view_frust_pts = np.linalg.inv(w2c) @ np.concatenate([view_frust_pts, np.ones_like(view_frust_pts[:1])], axis=0)
            view_frust_pts = view_frust_pts[:3]

            bnds[:, 0] = np.minimum(bnds[:, 0], view_frust_pts.min(axis=1))
            bnds[:, 1] = np.maximum(bnds[:, 1], view_frust_pts.max(axis=1))
        
        center = np.array(((bnds[0, 1] + bnds[0, 0]) / 2, (bnds[1, 1] + bnds[1, 0]) / 2,
                           (bnds[2, 1] + bnds[2, 0]) / 2)).astype(np.float32)

        lengths = bnds[:, 1] - bnds[:, 0]

        max_length = lengths.max(axis=0)
        radius = max_length / 2

        radius = radius * factor

        scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
        scale_mat[:3, 3] = center

        return scale_mat, 1. / radius
    
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sqrt(torch.sum(rays_d**2, dim=-1, keepdim=True))
        b = torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far
    
    def get_all_images(self):
        outputs = {
            "imgs": self.images.permute(0, 3, 1, 2),
            "c2ws": self.c2ws,
            "intrs":self.intrs
        }
        return outputs
    
    def get_random_rays(self, vid):
        vid = vid.item()
        
        pixels_x = torch.randint(low=0, high=self.img_hw[1], size=[self.n_rays])
        pixels_y = torch.randint(low=0, high=self.img_hw[0], size=[self.n_rays])
        
        color = self.images[vid][(pixels_y.long(), pixels_x.long())]    # n_rays, 3
        mask = self.masks[vid][(pixels_y.long(), pixels_x.long())]   # n_rays
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # n_rays, 3
        p = torch.matmul(self.intrs[vid].inverse()[None, :3, :3], p[:, :, None]).squeeze() # n_rays, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # n_rays, 3
        rays_d = torch.matmul(self.c2ws[vid, None, :3, :3], rays_d[:, :, None]).squeeze()  # n_rays, 3
        rays_o = self.c2ws[vid, None, :3, 3].expand(rays_d.shape) # n_rays, 3
        near, far = self.near_fars[vid].reshape(1, 2).split(split_size=1, dim=1)
        
        view_ids = [vid] + list(range(self.num_views))[:vid] + list(range(self.num_views))[vid+1:]
        intrs = self.intrs[view_ids]
        c2ws = self.c2ws[view_ids]
        imgs = self.images[view_ids].permute(0, 3, 1, 2)
        
        outputs = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "near": near,
            "far": far,
            "color": color,
            "intrs": intrs,
            "c2ws": c2ws,
            "view_ids": view_ids,
            "imgs": imgs,
        }
        
        return outputs
    
    def get_rays_at(self, vid):
        tx = torch.linspace(0, self.img_hw[1] - 1, self.img_hw[1] // self.val_res_level)
        ty = torch.linspace(0, self.img_hw[0] - 1, self.img_hw[0] // self.val_res_level)
        pixels_y, pixels_x = torch.meshgrid(ty, tx)
        pixels_x, pixels_y = pixels_x.reshape(-1), pixels_y.reshape(-1)
        
        color = self.images[vid][(pixels_y.long(), pixels_x.long())]    # n_rays, 3
        mask = self.masks[vid][(pixels_y.long(), pixels_x.long())]   # n_rays
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # n_rays, 3
        p = torch.matmul(self.intrs[vid].inverse()[None, :3, :3], p[:, :, None]).squeeze() # n_rays, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # n_rays, 3
        rays_d = torch.matmul(self.c2ws[vid, None, :3, :3], rays_d[:, :, None]).squeeze()  # n_rays, 3
        rays_o = self.c2ws[vid, None, :3, 3].expand(rays_d.shape) # n_rays, 3
        near, far = self.near_fars[vid].reshape(1, 2).split(split_size=1, dim=1)
        
        view_ids = [vid] + list(range(self.num_views))[:vid] + list(range(self.num_views))[vid+1:]
        intrs = self.intrs[view_ids]
        c2ws = self.c2ws[view_ids]
        imgs = self.images[view_ids].permute(0, 3, 1, 2)
        masks = self.masks[view_ids]
        bound_min=torch.tensor([-1, -1, -1], dtype=torch.float32)
        bound_max=torch.tensor([1, 1, 1], dtype=torch.float32)
        hw = torch.Tensor([self.img_hw[0]//self.val_res_level, self.img_hw[1]//self.val_res_level]).int()
        
        outputs = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "near": near,
            "far": far,
            "color": color,
            "intrs": intrs,
            "c2ws": c2ws,
            "view_ids": view_ids,
            "scale_mat": self.scale_mat,
            "scene": self.scene,
            "imgs": imgs,
            "masks": masks,
            "bound_min": bound_min,
            "bound_max": bound_max,
            "hw": hw
        }
        
        return outputs