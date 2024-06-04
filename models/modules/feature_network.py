import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=True, relu=True, **kwargs):
        super(Conv2d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=(not bn), **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    

class Conv2dAttn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=True, relu=True, **kwargs):
        super(Conv2dAttn, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=(not bn), **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        
        _, _, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + 1e-4)) + 0.5
        x = x * torch.sigmoid(y)
        
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    

class Deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=True, relu=True, **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FeatureNetworkOld(nn.Module):
    def __init__(self, confs):
        super(FeatureNetworkOld, self).__init__()
        d_base = confs.get_int("d_base")
        d_out = confs.get_int("d_out")
        self.conv0 = nn.Sequential(
            Conv2d(3, d_base, 3, 1),
            Conv2d(d_base, d_base, 3, 1)
        )
        self.conv1 = nn.Sequential(
            Conv2d(d_base, d_base*2, 5, 2),
            Conv2d(d_base*2, d_base*2, 3, 1),
            Conv2d(d_base*2, d_base*2, 3, 1)
        )
        self.conv2 = nn.Sequential(
            Conv2d(d_base*2, d_base*4, 5, 2),
            Conv2d(d_base*4, d_base*4, 3, 1),
            Conv2d(d_base*4, d_base*4, 3, 1)
        )
        
        self.out2 = nn.Conv2d(d_base*4, d_out, 3, 1, 1, bias=False)
        self.out1 = nn.Conv2d(d_base*4, d_out, 3, 1, 1, bias=False)
        self.out0 = nn.Conv2d(d_base*4, d_out, 3, 1, 1, bias=False)
        
        self.inner1 = nn.Conv2d(d_base*2, d_base*4, 3, 1, 1, bias=False)
        self.inner0 = nn.Conv2d(d_base, d_base*4, 3, 1, 1, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (nv, c, h, w)
        """
        
        feat0 = self.conv0(x)
        feat1 = self.conv1(feat0)
        feat2 = self.conv2(feat1)
        
        out2 = self.out2(feat2)
        intra_feat = F.interpolate(feat2, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(feat1)
        out1 = self.out1(intra_feat)
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner0(feat0)
        out0 = self.out0(intra_feat)
        
        outs = [out2, out1, out0]
        
        return outs
    
    
class FeatureNetwork(nn.Module):
    def __init__(self, confs):
        super(FeatureNetwork, self).__init__()

        d_in = confs.get_int("d_in")
        d_base = confs.get_int("d_base")
        d_outs = confs.get_list("d_out") #fine 2 coarse order
        num_stage =  len(d_outs)
        self.num_stage = num_stage

        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        self.out_layers = nn.ModuleList([])

        for i in range(num_stage):
            dim_m = d_base * 2**i
            stride = 2 if i > 0 else 1
            encod_layer = nn.Sequential(
                Conv2d(d_in, dim_m, 3, stride),
                Conv2d(dim_m, dim_m, 3, 1)
                # Conv2dAttn(dim_m, dim_m, 3, 1)
            )
            self.encoder_layers.append(encod_layer)
            d_in = dim_m

            out_layer = nn.Conv2d(dim_m, d_outs[i], 3, 1, 1, bias=False)
            self.out_layers.append(out_layer)

            if i < num_stage-1:
                decod_layer = Deconv2d(d_base * 2**(i+1), d_base * 2**i, stride=2, padding=1, output_padding=1)
                self.decoder_layers.append(decod_layer)

    def forward(self, x):

        e_outs = []

        for i in range(self.num_stage):
            e_out = self.encoder_layers[i](x)
            e_outs.append(e_out)
            x = e_out

        d_outs = [e_out]
        for i in range(self.num_stage-2, -1, -1):
            d_out = self.decoder_layers[i](d_outs[-1]) + e_outs[i]
            d_outs.append(d_out)
        d_outs = d_outs[::-1]

        outs = []
        for i in range(self.num_stage):
            out = self.out_layers[i](d_outs[i])
            outs.append(out)
        
        return outs[::-1]   # low_res 2 high_res
        