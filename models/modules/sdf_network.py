import torch
import torch.nn as nn
import numpy as np
import math

from .embedder import get_embedder
from .projector import lookup_volume


@torch.enable_grad()
def get_base_gradients(x, sdf_volume):
    x.requires_grad_(True)
    y = lookup_volume(x, sdf_volume)
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)

    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    
    return gradients


class SDFNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers, skip_in=(4,), multires=0, bias=0.5, scale=1, geometric_init=True,
                 weight_norm=True, inside_outside=False, feat_channels=32, feat_multires=2):
        super(SDFNetwork, self).__init__()
        
        self.init_feat_channels = feat_channels

        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            d_in = input_ch
        
        self.embed_fn_feat = None
        if feat_multires > 0:
            embed_fn, input_ch = get_embedder(feat_multires, input_dims=feat_channels)
            self.embed_fn_feat = embed_fn
            feat_channels = input_ch
            
        dims = [d_in] + [d_hidden + feat_channels for _ in range(n_layers)] + [d_out]

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l < self.num_layers - 2:
                out_dim = out_dim - feat_channels

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                    # the channels for latent codes are set to 0
                    torch.nn.init.constant_(lin.weight[:, -feat_channels:], 0.0)
                    torch.nn.init.constant_(lin.bias[-feat_channels:], 0.0)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    # torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3 + feat_channels):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    # the channels for latent code are initialized to 0
                    torch.nn.init.constant_(lin.weight[:, -feat_channels:], 0.0)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        

    def forward(self, inputs, volumes):
        
        feats = lookup_volume(inputs.clone(), volumes)
        if self.embed_fn_feat is not None:
            feats = self.embed_fn_feat(feats)

        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)
            
            if 0 < l < self.num_layers - 1:
                x = torch.cat([x, feats], -1)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, volumes):
        return self.forward(x, volumes)[:, :1]

    def sdf_hidden_appearance(self, x, volumes):
        return self.forward(x, volumes)
    
    @torch.enable_grad()
    def gradient(self, x, volumes):
        x.requires_grad_(True)
        y = self.sdf(x, volumes)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)

        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        d_output2 = torch.ones_like(gradients, requires_grad=False, device=gradients.device)
        smooth = torch.autograd.grad(
            outputs=gradients,
            inputs=x,
            grad_outputs=d_output2,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return gradients, smooth