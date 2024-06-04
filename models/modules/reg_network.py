import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=True, relu=True, **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.InstanceNorm3d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.relu = nn.ELU(inplace=True) if relu else None
        # self.relu = nn.Softplus() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Deconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=True, relu=True, **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.InstanceNorm3d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.relu = nn.ELU(inplace=True) if relu else None
        # self.relu = nn.Softplus() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    
class GlobalAttn(nn.Module):
    def __init__(self, d_model, d_ff, d_keys, d_values, n_heads, dropout=0.0):
        super(GlobalAttn, self).__init__()
        
        self.n_heads = n_heads
        self.d_keys = d_keys
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
    def forward(self, x):
        """ 
        x: (1, c, d, h, w)
        """
        B, c, d, h, w = x.shape
        x = x.reshape(B, c, -1).permute(0, 2, 1).contiguous()
        
        L = x.shape[1]
        H = self.n_heads
        
        Q = self.query_projection(x).reshape(B, L, H, -1).permute(0, 2, 1, 3).contiguous()
        K = self.key_projection(x).reshape(B, L, H, -1).permute(0, 2, 1, 3).contiguous()
        V = self.value_projection(x).reshape(B, L, H, -1).permute(0, 2, 1, 3).contiguous()
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) /  math.sqrt(self.d_keys)
        scores = F.softmax(scores, dim=-1)
        
        out = torch.matmul(scores, V)    # (B, H, L, dv)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, L, -1)
        out = self.out_projection(out)
        
        out = x + self.dropout(self.activation(out))
        out2 = self.norm1(out)
        out = self.dropout(self.activation(self.linear1(out2)))
        out = self.dropout(self.linear2(out))
        out = self.norm2(out + out2)
        
        out = out.reshape(B, d, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
        
        return out
        

class RegNetwork(nn.Module):
    def __init__(self, conf):
        super(RegNetwork, self).__init__()

        d_voluem = conf.get_list("d_voluem")
        d_base = conf.get_int("d_base")
        d_out = conf.get_list("d_out") # coarse 2 fine order
        num_stage =  len(d_out)
        self.num_stage = num_stage

        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        self.out_layers = nn.ModuleList([])

        self.conv0 = Conv3d(d_voluem[0], d_base, 3, 1, padding=1)
        d_in = d_base
        for i in range(num_stage):
            dim_m = d_base * 2**i
            stride = 2 #if i > 0 else 1
            encod_layer = nn.Sequential(
                Conv3d(d_in, dim_m, 3, stride, padding=1),
                Conv3d(dim_m, dim_m, 3, 1, padding=1)
            )
            self.encoder_layers.append(encod_layer)
            if i < num_stage-1:
                d_in = dim_m + d_voluem[i+1]

            out_layer = nn.Conv3d(d_base * 2**(max(i-1, 0)), d_out[i], 3, 1, 1)
            self.out_layers.append(out_layer)

            # if i < num_stage-1:
            decod_layer = Deconv3d(dim_m, d_base * 2**(max(i-1, 0)), stride=2, padding=1, output_padding=1)
            self.decoder_layers.append(decod_layer)
            
        # self.global_attn = GlobalAttn(d_base * 2**(num_stage-1), d_base * 4, d_base * 4, d_base * 4, 4, 0.2)

    def forward(self, volumes):
        
        assert len(volumes) == self.num_stage

        e_outs = []
        e_out = self.conv0(volumes[0])
        e_outs.append(e_out)
        for i in range(self.num_stage):
            e_out = self.encoder_layers[i](e_out)
            e_outs.append(e_out)
            if i < self.num_stage - 1:
                e_out = torch.cat([e_out, volumes[i+1]], dim=1)
                
        # e_out = self.global_attn(e_out)

        d_outs = [e_out]
        for i in range(self.num_stage-1, -1, -1):
            d_out = self.decoder_layers[i](d_outs[-1]) + e_outs[i]
            d_outs.append(d_out)
        d_outs = d_outs[::-1]

        outs = []
        for i in range(self.num_stage):
            out = self.out_layers[i](d_outs[i])
            outs.append(out)
        
        return outs   # high_res 2 low_res
    
    
class RegNetworkLite(nn.Module):
    def __init__(self, conf):
        super(RegNetworkLite, self).__init__()

        d_voluem = conf.get_list("d_voluem")
        d_base = conf.get_int("d_base")
        d_out = conf.get_int("d_out") 
        num_stage =  len(d_voluem)
        self.num_stage = num_stage

        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])

        self.conv0 = Conv3d(d_voluem[0], d_base, 3, 1, padding=1)
        d_in = d_base
        for i in range(num_stage):
            dim_m = d_base * 2**i
            stride = 2 #if i > 0 else 1
            encod_layer = nn.Sequential(
                Conv3d(d_in, dim_m, 3, stride, padding=1),
                Conv3d(dim_m, dim_m, 3, 1, padding=1)
            )
            self.encoder_layers.append(encod_layer)
            if i < num_stage-1:
                d_in = dim_m + d_voluem[i+1]

            # if i < num_stage-1:
            decod_layer = Deconv3d(dim_m, d_base * 2**(max(i-1, 0)), stride=2, padding=1, output_padding=1)
            self.decoder_layers.append(decod_layer)
        
        self.out_layer = nn.Conv3d(d_base, d_out, 3, 1, 1)

    def forward(self, volumes):
        
        assert len(volumes) == self.num_stage

        e_outs = []
        e_out = self.conv0(volumes[0])
        e_outs.append(e_out)
        for i in range(self.num_stage):
            e_out = self.encoder_layers[i](e_out)
            e_outs.append(e_out)
            if i < self.num_stage - 1:
                e_out = torch.cat([e_out, volumes[i+1]], dim=1)
                
        # e_out = self.global_attn(e_out)

        d_out = e_out
        for i in range(self.num_stage-1, -1, -1):
            d_out = self.decoder_layers[i](d_out) + e_outs[i]

        out = self.out_layer(d_out)
        
        return out
