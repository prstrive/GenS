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


class DTUDataset(Dataset):
    def __init__(self, confs, mode):
        super(DTUDataset, self).__init__()

        self.mode = mode
        self.data_dir = confs['data_dir']
        self.num_src_view = confs.get_int('num_src_view')
        self.interval_scale = confs.get_float('interval_scale')
        self.num_interval = confs.get_int('num_interval')
        self.img_hw = confs['img_hw']
        self.n_rays = confs.get_int('n_rays', 0)
        self.factor = confs.get_float('factor')
        self.total_views = 49

        self.split = confs.get_string("split", default=None)
        self.scene = confs.get_list('scene', default=None)
        self.light_idx = confs.get_list('light_idx', default=None)
        self.ref_view = confs.get_list('ref_view', default=None)
        if mode == "val":
            self.val_res_level = confs.get_int('val_res_level', default=1)

        self.intrs, self.w2cs, self.near_fars = self.read_cam_info()
        self.pairs = self.get_pairs()
        self.metas = self.build_list()
        
        self.pseudo_scale = 0.0037506045743823813

    def get_pairs(self, num_select=10):
        
        pair_file = "Cameras/pair.txt"
        if os.path.exists(os.path.join(self.data_dir, pair_file)):
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
        else:
            print("Calculating pair...")
            w2cs = np.stack(self.w2cs, axis=0)
            c2ws = np.linalg.inv(w2cs)
            dists = np.linalg.norm(c2ws[:, None, :3, 3] - c2ws[None, :, :3, 3], axis=-1)
            eyes = np.eye(dists.shape[0])
            dists[eyes>0] = 1e3
            sorted_vids = np.argsort(dists, axis=1)
            pairs = sorted_vids[:, :num_select]

        return pairs

    def build_list(self):
        metas = []

        if self.scene is not None:
            scans = self.scene
        else:
            if self.split is not None:
                with open(self.split) as f:
                    scans = f.readlines()
                    scans = [line.rstrip() for line in scans]
            else:
                raise ValueError("There are no scenes!")
        
        light_idxs = range(7)
        if self.light_idx is not None:
            light_idxs = self.light_idx

        # scans
        for scan in scans:
            # num_viewpoint = len(os.listdir(os.path.join(self.data_dir, 'Rectified/{}_train/'.format(scan)))) // 7
            num_viewpoint = self.total_views

            all_ref_views = [i for i in range(num_viewpoint)] if self.ref_view is None else self.ref_view

            for ref_view in all_ref_views:
                # pairs = list(self.pairs[ref_view])
                # src_views = pairs[:min(self.num_src_view, len(pairs))]
                
                # light conditions 0-6
                for light_idx in light_idxs:
                    metas.append((scan, light_idx, ref_view))
                        
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def read_cam_info(self):
        intrs = []
        w2cs = []
        near_fars = []
        for vid in range(self.total_views):
            filename = os.path.join(self.data_dir, 'Cameras/{:0>8}_cam.txt').format(vid)

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

            intrinsics_[0] *= self.img_hw[1] / 1600
            intrinsics_[1] *= self.img_hw[0] / 1200

            intrs.append(intrinsics_)
            w2cs.append(extrinsics)
            near_fars.append([depth_min, depth_max])
        
        return intrs, w2cs, near_fars

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

    def read_img(self, filename):
        # 1200, 1600
        img = np.array(Image.open(filename), dtype=np.float32)
        # # 600 800
        # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        # # 512, 640
        # img = img[44:556, 80:720]  

        img = cv2.resize(img, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST)
        return img
    
    def read_numpy(self, filename):
        # 1200, 1600
        img = np.load(filename).astype(np.float32)
        # # 600 800
        # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        # # 512, 640
        # img = img[44:556, 80:720]  

        img = cv2.resize(img, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST)
        return img
    
    def read_depth(self, filename):
        # 1200, 1600
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        # # 600 800
        # depth = cv2.resize(depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        # # 512, 640
        # depth = depth[44:556, 80:720]  

        depth = cv2.resize(depth, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return depth

    def __getitem__(self, idx):
        scan, light_idx, ref_view = self.metas[idx]
        pairs = list(self.pairs[ref_view])
        if self.mode == "train":
            src_views = random.sample(pairs[:6], min(self.num_src_view, len(pairs)))
            # src_views = pairs[:min(self.num_src_view, len(pairs))]
        else:
            src_views = pairs[:min(self.num_src_view, len(pairs))]
        view_ids = [ref_view] + src_views
        
        imgs = []
        intrs = []
        w2cs = []
        near_fars = []
        masks = []
        depths = []
        
        src_idx = np.random.randint(1, len(view_ids))
        
        w2c_ref = self.w2cs[ref_view]
        w2c_ref_inv = np.linalg.inv(w2c_ref)
        
        for i, vid in enumerate(view_ids):
            if vid > 48:
                img_filename = os.path.join(self.data_dir, 'Rectified_raw/{}/rect_{:0>3}_{}_r7000.png'.format(scan, vid + 1, light_idx))
            else:
                img_filename = os.path.join(self.data_dir, 'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            
            mask_filename = os.path.join(self.data_dir, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
                
            depth_filename = os.path.join(self.data_dir, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            
            pseudo_depth_filename = os.path.join(self.data_dir, 'pseudo_depths/{}/{}_epoch0.npy'.format(scan, vid))
            
            img = self.read_img(img_filename) / 256.0
            intr, w2c, near_far = self.intrs[vid], self.w2cs[vid], self.near_fars[vid]
            mask = (self.read_img(mask_filename) > 10).astype(np.float32)
            if(len(mask.shape)>2):
                mask = (np.mean(mask, axis=-1) > 0).astype(np.float32)
            depth = self.read_depth(depth_filename)
        
            imgs.append(img)
            intrs.append(intr)
            w2cs.append(w2c @ w2c_ref_inv)
            near_fars.append(near_far)
            masks.append(mask)
            depths.append(depth)
            
            if i==0:
                ref_pseudo_depth = self.read_numpy(pseudo_depth_filename) / self.pseudo_scale if self.mode=="train" else masks[0]
        
        scale_mat, scale_factor = self.get_scale_mat(self.img_hw, intrs, w2cs, near_fars, factor=self.factor)
        
        c2ws = []
        new_near_fars = []
        new_intrs = []
        new_depths = []
        for intr, w2c, depth in zip(intrs, w2cs, depths):
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
            new_depths.append(scale_factor * depth)
            
        ref_pseudo_depth = ref_pseudo_depth * scale_factor
        ref_pseudo_depth = torch.from_numpy(ref_pseudo_depth.astype(np.float32))

        imgs = torch.from_numpy(np.stack(imgs).astype(np.float32))
        intrs = torch.from_numpy(np.stack(new_intrs).astype(np.float32))
        c2ws = torch.from_numpy(np.stack(c2ws).astype(np.float32))
        near_fars = torch.from_numpy(np.stack(new_near_fars).astype(np.float32))
        masks = torch.from_numpy(np.stack(masks).astype(np.float32))
        depths = torch.from_numpy(np.stack(new_depths).astype(np.float32))
        
        outputs = {
            "imgs": imgs.permute(0, 3, 1, 2).contiguous(),
            "intrs": intrs,
            "c2ws": c2ws,
            "masks": masks,
            "scale_mat": torch.from_numpy(w2c_ref_inv @ scale_mat),
            "view_ids": torch.from_numpy(np.array(view_ids)).long()
        }
        
        ys, xs = torch.meshgrid(torch.linspace(0, self.img_hw[0] - 1, self.img_hw[0]),
                                torch.linspace(0, self.img_hw[1] - 1, self.img_hw[1]))  # pytorch's meshgrid has indexing='ij'
        pixel_all = torch.stack([xs, ys], dim=-1)  # H, W, 2

        if self.mode == "train":
            assert self.n_rays>0, "No sampling rays!"
            
            ref_n_rays = self.n_rays

            p_valid = pixel_all[masks[0] > 0.5]  # [num, 2]
            pixels_x_i = torch.randint(low=0, high=self.img_hw[1], size=[ref_n_rays // 4])
            pixels_y_i = torch.randint(low=0, high=self.img_hw[0], size=[ref_n_rays // 4])
            random_idx = torch.randint(low=0, high=p_valid.shape[0], size=[ref_n_rays - ref_n_rays // 4])
            p_select = p_valid[random_idx]
            pixels_x = p_select[:, 0]
            pixels_y = p_select[:, 1]

            pixels_x = torch.cat([pixels_x, pixels_x_i], dim=0)
            pixels_y = torch.cat([pixels_y, pixels_y_i], dim=0)

        else:
            bound_min=torch.tensor([-1, -1, -1], dtype=torch.float32)
            bound_max=torch.tensor([1, 1, 1], dtype=torch.float32)
            outputs.update({"bound_min": bound_min, "bound_max": bound_max, "scene": scan})
            outputs["file_name"] = scan+"_view"+str(ref_view)+"_light"+str(light_idx)
            outputs["hw"] = torch.Tensor([self.img_hw[0]//self.val_res_level, self.img_hw[1]//self.val_res_level]).int()

            tx = torch.linspace(0, self.img_hw[1] - 1, self.img_hw[1] // self.val_res_level)
            ty = torch.linspace(0, self.img_hw[0] - 1, self.img_hw[0] // self.val_res_level)
            pixels_y, pixels_x = torch.meshgrid(ty, tx)
            pixels_x, pixels_y = pixels_x.reshape(-1), pixels_y.reshape(-1)
        
        pseudo_depth = ref_pseudo_depth[(pixels_y.long(), pixels_x.long())]   # n_rays
        color = imgs[0][(pixels_y.long(), pixels_x.long())]    # n_rays, 3
        depth = depths[0][(pixels_y.long(), pixels_x.long())]   # n_rays
        mask = masks[0][(pixels_y.long(), pixels_x.long())]   # n_rays
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # n_rays, 3
        p = torch.matmul(intrs.inverse()[0, None, :3, :3], p[:, :, None]).squeeze() # n_rays, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # n_rays, 3
        rays_d = torch.matmul(c2ws[0, None, :3, :3], rays_d[:, :, None]).squeeze()  # n_rays, 3
        rays_o = c2ws[0, None, :3, 3].expand(rays_d.shape) # n_rays, 3
        near, far = near_fars[0].reshape(1, 2).split(split_size=1, dim=1)
        
        p_mask = (ref_pseudo_depth > 0) & (masks[0] > 0)
        if (self.mode=="train") and (p_mask.sum() > 100):
            y, x = torch.meshgrid(torch.arange(0, self.img_hw[0]), torch.arange(0, self.img_hw[1]))
            x = x[p_mask].type_as(intrs)
            y = y[p_mask].type_as(intrs)
            p_depth = ref_pseudo_depth[p_mask]
            random_idx = torch.randint(low=0, high=x.shape[0], size=[2048])
            x = x[random_idx]
            y = y[random_idx]
            p_depth = p_depth[random_idx]
            xyz_ref = torch.matmul(intrs.inverse()[0, :3, :3], torch.stack((x, y, torch.ones_like(x)), dim=0) * p_depth.unsqueeze(0))
            xyz_world = torch.matmul(c2ws[0], torch.cat((xyz_ref, torch.ones_like(x).unsqueeze(0)), dim=0))[:3]
            pseudo_pts = xyz_world.permute(1, 0)
            outputs["pseudo_pts"] = pseudo_pts
        
        outputs.update({
            "rays_o": rays_o,
            "rays_d": rays_d,
            "near": near,
            "far": far,
            "color": color,
            "depth": depth,
            "pseudo_depth": pseudo_depth,
            "depth_ref": depths[0],
            "mask": mask,
            "mask_ref": masks[0],
            "depth_ref": depths[0],
            "pseudo_depth_ref": ref_pseudo_depth,
            "src_idx": src_idx,
        })

        return outputs
    
    def __len__(self):
        return len(self.metas)