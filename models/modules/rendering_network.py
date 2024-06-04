import torch
import torch.nn as nn
import numpy as np

from .embedder import get_embedder
    
    
# # This implementation is borrowed from IDR: https://github.com/lioryariv/idr
# class RenderingNetwork(nn.Module):
#     def __init__(self,
#                  d_feature,
#                  mode,
#                  d_in,
#                  d_out,
#                  d_hidden,
#                  n_layers,
#                  skip_in=(2,),
#                  weight_norm=True,
#                  multires_view=0,
#                  squeeze_out=True):
#         super(RenderingNetwork, self).__init__()

#         self.mode = mode
#         self.squeeze_out = squeeze_out
#         dims = [d_in] + [d_hidden + d_feature for _ in range(n_layers)] + [d_out]

#         self.embedview_fn = None
#         if multires_view > 0:
#             embedview_fn, input_ch = get_embedder(multires_view)
#             self.embedview_fn = embedview_fn
#             dims[0] += (input_ch - 3)

#         self.num_layers = len(dims)
#         self.skip_in = skip_in

#         for l in range(0, self.num_layers - 1):
#             if l + 1 in self.skip_in:
#                 out_dim = dims[l + 1] - dims[0]
#             else:
#                 out_dim = dims[l + 1]
            
#             if l < self.num_layers - 2:
#                 out_dim = out_dim - d_feature
            
#             lin = nn.Linear(dims[l], out_dim)

#             if weight_norm:
#                 lin = nn.utils.weight_norm(lin)

#             setattr(self, "lin" + str(l), lin)

#         self.relu = nn.ReLU()

#     def forward(self, points, normals, view_dirs, feature_vectors):
#         if self.embedview_fn is not None:
#             view_dirs = self.embedview_fn(view_dirs)

#         rendering_input = None

#         if self.mode == 'idr':
#             rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
#             # rendering_input = torch.cat([view_dirs, normals, feature_vectors], dim=-1)
#         elif self.mode == 'no_view_dir':
#             rendering_input = torch.cat([points, normals], dim=-1)
#         elif self.mode == 'no_normal':
#             rendering_input = torch.cat([points, view_dirs], dim=-1)

#         x = rendering_input

#         for l in range(0, self.num_layers - 1):
#             lin = getattr(self, "lin" + str(l))
            
#             if l in self.skip_in:
#                 x = torch.cat([x, rendering_input], -1) / np.sqrt(2)
            
#             if 0 < l < self.num_layers - 1:
#                 x = torch.cat([x, feature_vectors], -1)

#             x = lin(x)

#             if l < self.num_layers - 2:
#                 x = self.relu(x)

#         if self.squeeze_out:
#             x = torch.sigmoid(x)
#         return x
    
    
# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x