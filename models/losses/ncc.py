import torch
import torch.nn.functional as F
import numpy as np
from math import exp, sqrt


def compute_LNCC(ref_gray, src_grays):
    # ref_gray: [1, batch_size, 121, n]
    # src_grays: [nsrc, batch_size, 121, n]
    ref_gray = ref_gray.permute(1, 0, 3, 2).contiguous()  # [batch_size, 1, n, 121]
    src_grays = src_grays.permute(1, 0, 3, 2).contiguous()  # [batch_size, nsrc, n, 121]

    ref_src = ref_gray * src_grays  # [batch_size, nsrc, n, npatch]

    bs, nsrc, nc, npatch = src_grays.shape
    patch_size = int(sqrt(npatch))
    ref_gray = ref_gray.view(bs, 1, nc, patch_size, patch_size).view(-1, nc, patch_size, patch_size)
    src_grays = src_grays.view(bs, nsrc, nc, patch_size, patch_size).contiguous().view(-1, nc, patch_size, patch_size)
    ref_src = ref_src.view(bs, nsrc, nc, patch_size, patch_size).contiguous().view(-1, nc, patch_size, patch_size)

    ref_sq = ref_gray.pow(2)
    src_sq = src_grays.pow(2)

    filters = torch.ones(nc, 1, patch_size, patch_size, device=ref_gray.device)
    # filters = create_window(patch_size, nc, std=3).type_as(ref_gray) * 10
    # print("filters:", filters[0, 0])
    padding = patch_size // 2

    ref_sum = F.conv2d(ref_gray, filters, stride=1, padding=padding, groups=nc)[:, :, padding, padding].view(bs, 1, nc)
    src_sum = F.conv2d(src_grays, filters, stride=1, padding=padding, groups=nc)[:, :, padding, padding].view(bs, nsrc, nc)
    ref_sq_sum = F.conv2d(ref_sq, filters, stride=1, padding=padding, groups=nc)[:, :, padding, padding].view(bs, 1, nc)
    src_sq_sum = F.conv2d(src_sq, filters, stride=1, padding=padding, groups=nc)[:, :, padding, padding].view(bs, nsrc, nc)
    ref_src_sum = F.conv2d(ref_src, filters, stride=1, padding=padding, groups=nc)[:, :, padding, padding].view(bs, nsrc, nc)

    u_ref = ref_sum / npatch
    u_src = src_sum / npatch

    cross = ref_src_sum - u_src * ref_sum - u_ref * src_sum + u_ref * u_src * npatch
    ref_var = ref_sq_sum - 2 * u_ref * ref_sum + u_ref * u_ref * npatch
    src_var = src_sq_sum - 2 * u_src * src_sum + u_src * u_src * npatch

    cc = cross * cross / (ref_var * src_var + 1e-5)  # [batch_size, nsrc, nc]
    ncc = 1 - cc
    # ncc, _ = torch.clamp(ncc, 0.0, 2.0).topk(k=4, dim=2, largest=False)
    # ncc = ncc.mean(dim=2)
    # ncc, _ = torch.clamp(ncc, 0.0, 2.0).min(dim=2)
    ncc = torch.clamp(ncc, 0.0, 2.0).mean(dim=2)
    ncc, _ = torch.topk(ncc, 2, dim=1, largest=False)
    ncc = torch.mean(ncc, dim=1, keepdim=True)  # [batch_size, 1]
    
    return ncc