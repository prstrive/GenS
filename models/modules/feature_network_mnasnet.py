import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
    

class Deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=True, relu=True, **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels) if bn else None
        # self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.relu = nn.ELU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FeatureNetwork(nn.Module):
    def __init__(self, confs):
        super(FeatureNetwork, self).__init__()
        d_out = confs.get_list("d_out")
        
        backbone_mobile_layers = list(models.mnasnet1_0(pretrained=True).layers.children())
        
        self.layer1 = torch.nn.Sequential(*backbone_mobile_layers[0:8])
        self.layer2 = torch.nn.Sequential(*backbone_mobile_layers[8:9])
        self.layer3 = torch.nn.Sequential(*backbone_mobile_layers[9:10])
        self.layer4 = torch.nn.Sequential(*backbone_mobile_layers[10:12])
        self.layer5 = torch.nn.Sequential(*backbone_mobile_layers[12:14])
        
        self.decod_layer5 = Deconv2d(320, 96, stride=2, padding=1, output_padding=1)
        self.decod_layer4 = Deconv2d(96, 40, stride=2, padding=1, output_padding=1)
        self.decod_layer3 = Deconv2d(40, 24, stride=2, padding=1, output_padding=1)
        self.decod_layer2 = Deconv2d(24, 16, stride=2, padding=1, output_padding=1)
        self.decod_layer1 = Deconv2d(16, 8, stride=2, padding=1, output_padding=1)
        
        self.out_layer5 = nn.Conv2d(96, d_out[4], 3, 1, 1, bias=False)
        self.out_layer4 = nn.Conv2d(40, d_out[3], 3, 1, 1, bias=False)
        self.out_layer3 = nn.Conv2d(24, d_out[2], 3, 1, 1, bias=False)
        self.out_layer2 = nn.Conv2d(16, d_out[1], 3, 1, 1, bias=False)
        self.out_layer1 = nn.Conv2d(8, d_out[0], 3, 1, 1, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: (nv, c, h, w)
        """
    
        enc1 = self.layer1(x)
        enc2 = self.layer2(enc1)
        enc3 = self.layer3(enc2)
        enc4 = self.layer4(enc3)
        enc5 = self.layer5(enc4)
        
        dec5 = self.decod_layer5(enc5) + enc4
        dec4 = self.decod_layer4(dec5) + enc3
        dec3 = self.decod_layer3(dec4) + enc2
        dec2 = self.decod_layer2(dec3) + enc1
        dec1 = self.decod_layer1(dec2)
        
        out5 = self.out_layer5(dec5)
        out4 = self.out_layer4(dec4)
        out3 = self.out_layer3(dec3)
        out2 = self.out_layer2(dec2)
        out1 = self.out_layer1(dec1)
        
        outs = [out1, out2, out3, out4, out5]   # fine to coarse
        
        return outs
        