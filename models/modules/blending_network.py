import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedder import get_embedder

# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=1, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=1, keepdim=True)
    return mean, var


class BlendingNetwork(nn.Module):
    """
    This model is not sensitive to finetuning
    """

    def __init__(self, d_feature=16, anti_alias_pooling=True):
        super(BlendingNetwork, self).__init__()

        self.anti_alias_pooling = anti_alias_pooling

        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)

        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, d_feature + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((d_feature + 3) * 3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.rgb_fc = nn.Sequential(nn.Linear(32 + 1 + 4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def forward(self, rgb_feat, ray_diff, mask):
        '''
        rgb_feat: (n_rays*n_samples, n_srcs, c+3)
        ray_diff: (n_rays*n_samples, n_srcs, 4)
        mask: (n_rays*n_samples, n_srcs)

        :param geometry_feat: geometry features indicates sdf  [n_rays, n_samples, n_feat]
        :param rgb_feat: rgbs and image features [n_views, n_rays, n_samples, n_feat]
        :param ray_diff: ray direction difference [n_views, n_rays, n_samples, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_views, n_rays, n_samples]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        mask = mask[:, :, None]
        num_views = rgb_feat.shape[1]

        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat

        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=1, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays*n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays*n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, num_views, -1), rgb_feat], dim=-1)  # [n_rays*n_samples, n_views, 3*n_feat+n_geo_feat]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
        vis = torch.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=1)  # color blending
        rgb_out = torch.sum(rgb_in * blending_weights_valid, dim=1)

        return rgb_out  # (N_rays, n_samples, 3)