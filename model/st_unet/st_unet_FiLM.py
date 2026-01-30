# This module stores the code for the CNN U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, ci_dim, n_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ci_dim, n_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels * 2, n_channels * 2)
        )

    def forward(self, x, ci):
        gamma_beta = self.mlp(ci)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with 3D Convolution"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, drop_channels=True, p_drop=None, ci_dim=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3D convolution layer added here with a smaller temporal kernel size
        self.conv3d = nn.Conv3d(mid_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size//2, kernel_size//2), bias=False)
        self.film = FiLM(ci_dim, mid_channels) if ci_dim is not None else None
        
        if drop_channels:
            self.double_conv.add_module('dropout', nn.Dropout2d(p=p_drop))

    def forward(self, x, ci=None):
        x = self.double_conv(x)  # 2D convolution
        if self.film is not None:
            if ci is None:
                raise ValueError("ci must be provided when ci_dim is set.")
            x = self.film(x, ci)
        # reshape input for 3D convolution (batch_size, mid_channels, depth = 1, height, width)
        x = x.unsqueeze(2)
        x = self.conv3d(x)  # 3D convolution
        x = x.squeeze(2)  # reshape to original dimension by removing the depth
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, pooling='max', drop_channels=False, p_drop=None, ci_dim=None):
        super().__init__()
        if pooling == 'max':
            self.pooling = nn.MaxPool2d(2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop, ci_dim=ci_dim)

    def forward(self, x, ci=None):
        x = self.pooling(x)
        return self.conv(x, ci=ci)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, bilinear=True, drop_channels=False, p_drop=None):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, init_hid_dim=8, kernel_size=3, pooling='max', bilinear=False, drop_channels=False, p_drop=None, ci_dim=None):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_hid_dim = init_hid_dim 
        self.bilinear = bilinear
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.drop_channels = drop_channels
        self.p_drop = p_drop
        self.ci_dim = ci_dim

        hid_dims = [init_hid_dim * (2**i) for i in range(5)]
        self.hid_dims = hid_dims

        # initial 2D Convolution (FiLM conditioning if ci_dim is provided)
        self.inc = DoubleConv(n_channels, hid_dims[0], kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop, ci_dim=ci_dim)

        # downscaling with 2D Convolution followed by pooling
        self.down1 = Down(hid_dims[0], hid_dims[1], kernel_size, pooling, drop_channels, p_drop, ci_dim=ci_dim)
        self.down2 = Down(hid_dims[1], hid_dims[2], kernel_size, pooling, drop_channels, p_drop, ci_dim=ci_dim)
        self.down3 = Down(hid_dims[2], hid_dims[3], kernel_size, pooling, drop_channels, p_drop, ci_dim=ci_dim)

        # temporal convolution with 3D Convolution
        self.temporal_conv = nn.Conv3d(hid_dims[3], hid_dims[3], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # downscaling with 2D Convolution followed by pooling
        factor = 2 if bilinear else 1
        self.bottleneck_channels = hid_dims[4] // factor
        self.down4 = Down(hid_dims[3], self.bottleneck_channels, kernel_size, pooling, drop_channels, p_drop, ci_dim=ci_dim)

        # upscaling with 2D Convolution followed by Double Convolution
        self.up1 = Up(hid_dims[4], hid_dims[3] // factor, kernel_size, bilinear, drop_channels, p_drop)
        self.up2 = Up(hid_dims[3], hid_dims[2] // factor, kernel_size, bilinear, drop_channels, p_drop)
        self.up3 = Up(hid_dims[2], hid_dims[1] // factor, kernel_size, bilinear, drop_channels, p_drop)

        # final 2D Convolution for output
        self.up4 = Up(hid_dims[1], hid_dims[0], kernel_size, bilinear, drop_channels, p_drop)
        self.outc = OutConv(hid_dims[0], n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ci=None):
        if self.ci_dim is not None:
            if ci is None:
                ci = x.new_zeros(x.size(0), self.ci_dim)
            if ci.dim() > 2:
                ci = ci.view(ci.size(0), -1)
            if ci.size(1) != self.ci_dim:
                raise ValueError("ci does not match ci_dim used to build the model.")
        elif ci is not None:
            raise ValueError("ci was provided but ci_dim was not set when building the model.")

        x1 = self.inc(x, ci=ci)
        x2 = self.down1(x1, ci=ci)
        x3 = self.down2(x2, ci=ci)
        x4 = self.down3(x3, ci=ci)

        # temporal 3D Convolution
        x4_temporal = x4.unsqueeze(2)  # add temporal dimension
        x4_temporal = self.temporal_conv(x4_temporal)
        x4 = x4_temporal.squeeze(2)  # remove temporal dimension

        x5 = self.down4(x4, ci=ci)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.sigmoid(x)
        return x
