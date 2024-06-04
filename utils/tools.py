import torch
import numpy as np
from skimage import measure
import random


def clean_volume(mask_volume):
    """ 
    mask_volume: (w, h, d)
    """
    label, num = measure.label(mask_volume, connectivity=3, return_num=True)
    print("Num region:", num)
    if num < 1:
        return mask_volume
    
    region = measure.regionprops(label)
    area_list = [region[i].area for i in range(num)]
    large_region_idx = np.argmax(area_list)
    for i in range(num):
        if i !=large_region_idx:
            label[region[i].slice][region[i].image] = 0
            
    return label


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def clean_volume(mask_volume):
    """ 
    mask_volume: (w, h, d)
    """
    label, num = measure.label(mask_volume, connectivity=3, return_num=True)
    print("Num region:", num)
    if num < 1:
        return mask_volume
    
    region = measure.regionprops(label)
    area_list = [region[i].area for i in range(num)]
    large_region_idx = np.argmax(area_list)
    for i in range(num):
        if i !=large_region_idx:
            label[region[i].slice][region[i].image] = 0
            
    return label


def save_depth(depth, file_path):
    import matplotlib as mpl
    import matplotlib.cm as cm
    from PIL import Image
    
    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)
    im.save(file_path)
    
    
# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, int):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def tensor2float(ipts):
    for k, v in ipts.items():
        if isinstance(v, torch.Tensor):
            ipts[k] = v.data.item()
    return ipts


def tensor2numpy(ipts):
    for k, v in ipts.items():
        if isinstance(v, np.ndarray):
            ipts[k] = v
        elif isinstance(v, torch.Tensor):
            ipts[k] = v.detach().cpu().numpy().copy()
    return ipts


def save_scalars(logger, mode, scalar_dict, global_step):
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.sum_data = {}
        self.avg_data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.sum_data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.sum_data[k] = v
                self.avg_data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.sum_data[k] += v
                self.avg_data[k] = self.sum_data[k] / self.count