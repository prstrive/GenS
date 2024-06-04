import os
import re
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


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


class BMVSDataset(Dataset):
    def __init__(self, confs, mode):
        super(BMVSDataset, self).__init__()
        
        self.mode = mode
        self.data_dir = confs['data_dir']
        self.num_src_view = confs.get_int('num_src_view')
        self.interval_scale = confs.get_float('interval_scale')
        self.num_interval = confs.get_int('num_interval')
        self.img_hw = confs['img_hw']
        self.n_rays = confs.get_int('n_rays', 0)
        self.factor = confs.get_float('factor')
        
        self.split = confs.get_string("split", default=None)
        self.scene = confs.get_list('scene', default=None)
        self.ref_view = confs.get_list('ref_view', default=None)
        self.src_views = confs.get_list('src_views', default=None)
        if mode == "val":
            self.val_res_level = confs.get_int('val_res_level', default=1)
            
        if self.scene is None:
            if self.split is not None:
                with open(self.split) as f:
                    scans = f.readlines()
                    self.scene = [line.rstrip() for line in scans]
            else:
                raise ValueError("There are no scenes!")
        
        self.metas = self.build_list()
            
    def build_list(self):
        metas = []
        
        for scene in self.scene:
            scene_path = os.path.join(self.data_dir, scene)
            pair_file = os.path.join(scene_path, "cams", "pair.txt")
            
            with open(pair_file) as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
            
            num_viewpoint = int(lines[0])
            all_ref_views = [i for i in range(num_viewpoint)] if self.ref_view is None else self.ref_view
            
            for ref_view in all_ref_views:
                if self.src_views is not None:
                    src_views = self.src_views
                else:
                    cluster_info = lines[2 * ref_view + 2].rstrip().split()
                    src_views = [int(x) for x in cluster_info[1::2]]
                metas.append((scene, ref_view, src_views))
                
        print("dataset", self.mode, "metas:", len(metas))
        return metas
    
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
    
    def read_cam(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        # depth_sample_num = float(lines[11].split()[2])
        # depth_max = float(lines[11].split()[3])
        depth_max = depth_min + depth_interval * self.num_interval
        
        intrinsics_[0] *= self.img_hw[1] / 768
        intrinsics_[1] *= self.img_hw[0] / 576
        
        return intrinsics_, extrinsics, [depth_min, depth_max]
    
    def read_img(self, filename):
        # 1200, 1600
        img = np.array(Image.open(filename), dtype=np.float32)
        # # 600 800
        # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        # # 512, 640
        # img = img[44:556, 80:720]  

        img = cv2.resize(img, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST)
        return img
    
    def read_depth_and_mask(self, filename, depth_min):
        # read pfm depth file
        # (576, 768)
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        mask = np.array(depth >= depth_min, dtype=np.float32)

        depth = cv2.resize(depth, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST)
        
        return depth, mask
    
    def __getitem__(self, idx):
        scan, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.num_src_view]
        imgs = []
        intrs = []
        w2cs = []
        near_fars = []
        depths = []
        masks = []
        
        for i, vid in enumerate(view_ids):
            img_path = os.path.join(self.data_dir, scan, 'blended_images', '%08d_masked.jpg' % vid)
            cam_path = os.path.join(self.data_dir, scan, 'cams', '%08d_cam.txt' % vid)

            img = self.read_img(img_path) / 256.0
            intr, w2c, near_far = self.read_cam(cam_path)
            
            imgs.append(img)
            intrs.append(intr)
            w2cs.append(w2c)
            near_fars.append(near_far)
            
            depth_path = os.path.join(self.data_dir, scan, 'rendered_depth_maps', '%08d.pfm' % vid)
            depth, mask = self.read_depth_and_mask(depth_path, near_far[0])
            depths.append(depth)
            masks.append(mask)
            
        w2c_ref = w2cs[0]
        w2c_ref_inv = np.linalg.inv(w2c_ref)
        new_w2cs = []
        for w2c in w2cs:
            new_w2cs.append(w2c @ w2c_ref_inv)
        w2cs = new_w2cs
        
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
            new_depths.append(depth * scale_factor)
            
        depths = torch.from_numpy(np.stack(new_depths).astype(np.float32))
        masks = torch.from_numpy(np.stack(masks).astype(np.float32))
            
        imgs = torch.from_numpy(np.stack(imgs).astype(np.float32))
        intrs = torch.from_numpy(np.stack(new_intrs).astype(np.float32))
        c2ws = torch.from_numpy(np.stack(c2ws).astype(np.float32))
        near_fars = torch.from_numpy(np.stack(new_near_fars).astype(np.float32))
        
        outputs = {
            "imgs": imgs.permute(0, 3, 1, 2).contiguous(),
            "intrs": intrs,
            "c2ws": c2ws,
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
            outputs["file_name"] = scan+"_view"+str(ref_view)
            outputs["hw"] = torch.Tensor([self.img_hw[0]//self.val_res_level, self.img_hw[1]//self.val_res_level]).int()
            outputs["masks"] = masks

            tx = torch.linspace(0, self.img_hw[1] - 1, self.img_hw[1] // self.val_res_level)
            ty = torch.linspace(0, self.img_hw[0] - 1, self.img_hw[0] // self.val_res_level)
            pixels_y, pixels_x = torch.meshgrid(ty, tx)
            pixels_x, pixels_y = pixels_x.reshape(-1), pixels_y.reshape(-1)
        
        color = imgs[0][(pixels_y.long(), pixels_x.long())]    # n_rays, 3
        depth = depths[0][(pixels_y.long(), pixels_x.long())]   # n_rays
        mask = masks[0][(pixels_y.long(), pixels_x.long())]   # n_rays
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # n_rays, 3
        p = torch.matmul(intrs.inverse()[0, None, :3, :3], p[:, :, None]).squeeze() # n_rays, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # n_rays, 3
        rays_d = torch.matmul(c2ws[0, None, :3, :3], rays_d[:, :, None]).squeeze()  # n_rays, 3
        rays_o = c2ws[0, None, :3, 3].expand(rays_d.shape) # n_rays, 3
        near, far = near_fars[0].reshape(1, 2).split(split_size=1, dim=1)
        
        outputs.update({
            "pixels_x": pixels_x,
            "pixels_y": pixels_y,
            "near_fars": near_fars,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "near": near,
            "far": far,
            "color": color,
            "depth": depth,
            "mask": mask,
            "masks": masks,
            "depth_ref": depths[0],
            "src_idx": 1
        })
        
        return outputs
    
    def __len__(self):
        return len(self.metas)