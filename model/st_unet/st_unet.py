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
        if ci.dim() == 2:
            gamma_beta = self.mlp(ci)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            return gamma * x + beta

        if ci.dim() != 3:
            raise ValueError("ci must be 2D (N, ci_dim) or 3D (N, D, ci_dim).")

        n, d, _ = ci.shape
        target_d = x.size(2)
        if d != target_d:
            # Resize CI along depth to match current feature depth
            ci_t = ci.permute(0, 2, 1)  # N, ci_dim, D
            ci_t = F.interpolate(ci_t, size=target_d, mode="linear", align_corners=False)
            ci = ci_t.permute(0, 2, 1)
            n, d, _ = ci.shape
        gamma_beta = self.mlp(ci.reshape(n * d, -1))
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(n, d, -1).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.view(n, d, -1).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with 3D Convolution"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, drop_channels=True, p_drop=None, ci_dim=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.film = FiLM(ci_dim, mid_channels) if ci_dim is not None else None
        self.dropout = nn.Dropout3d(p=p_drop) if drop_channels else None

    def forward(self, x, ci=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.film is not None:
            if ci is None:
                raise ValueError("ci must be provided when ci_dim is set.")
            x = self.film(x, ci)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, pooling='max', drop_channels=False, p_drop=None, ci_dim=None, down_time=False):
        super().__init__()
        if pooling == 'max':
            pool = (2, 2, 2) if down_time else (1, 2, 2)
            self.pooling = nn.MaxPool3d(pool)
        elif pooling == 'avg':
            pool = (2, 2, 2) if down_time else (1, 2, 2)
            self.pooling = nn.AvgPool3d(pool)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop, ci_dim=ci_dim)

    def forward(self, x, ci=None):
        x = self.pooling(x)
        return self.conv(x, ci=ci)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, drop_channels=False, p_drop=None, up_time=False, ci_dim=None):
        super().__init__()
        k = (2, 2, 2) if up_time else (1, 2, 2)
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=k, stride=k)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop, ci_dim=ci_dim)

    def forward(self, x1, x2, ci=None):
        x1 = self.up(x1)
        dT = x2.size(2) - x1.size(2)
        dH = x2.size(3) - x1.size(3)
        dW = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [dW // 2, dW - dW // 2,
                        dH // 2, dH - dH // 2,
                        dT // 2, dT - dT // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, ci=ci)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

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

        # initial 3D Convolution (FiLM conditioning if ci_dim is provided)
        # We treat n_channels as the number of frames (depth), so the input channel is 1.
        self.inc = DoubleConv(1, hid_dims[0], kernel_size=kernel_size, drop_channels=drop_channels, p_drop=p_drop, ci_dim=ci_dim)

        # Encoder: temporal downsampling for the first two blocks
        self.down1 = Down(hid_dims[0], hid_dims[1], kernel_size, pooling, drop_channels, p_drop, ci_dim=ci_dim, down_time=True)
        self.down2 = Down(hid_dims[1], hid_dims[2], kernel_size, pooling, drop_channels, p_drop, ci_dim=ci_dim, down_time=True)
        self.down3 = Down(hid_dims[2], hid_dims[3], kernel_size, pooling, drop_channels, p_drop, ci_dim=ci_dim, down_time=False)
        self.down4 = Down(hid_dims[3], hid_dims[4], kernel_size, pooling, drop_channels, p_drop, ci_dim=ci_dim, down_time=False)

        self.temporal = nn.Sequential(
            nn.Conv3d(hid_dims[4], hid_dims[4], (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(hid_dims[4]),
            nn.ReLU(inplace=True),
            nn.Conv3d(hid_dims[4], hid_dims[4], (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(hid_dims[4]),
            nn.ReLU(inplace=True),
        )

        # Decoder: mirror the encoder temporal upsampling
        self.up1 = Up(hid_dims[4], hid_dims[3], kernel_size, drop_channels, p_drop, up_time=False, ci_dim=ci_dim)
        self.up2 = Up(hid_dims[3], hid_dims[2], kernel_size, drop_channels, p_drop, up_time=False, ci_dim=ci_dim)
        self.up3 = Up(hid_dims[2], hid_dims[1], kernel_size, drop_channels, p_drop, up_time=True, ci_dim=ci_dim)
        self.up4 = Up(hid_dims[1], hid_dims[0], kernel_size, drop_channels, p_drop, up_time=True, ci_dim=ci_dim)

        self.outc = OutConv(hid_dims[0], n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ci=None):
        if x.dim() == 4:
            if x.size(1) != self.n_channels:
                raise ValueError(
                    f"Input has {x.size(1)} channels but model expects {self.n_channels}."
                )
            x = x.unsqueeze(1)
        elif x.dim() == 5:
            if x.size(1) == 1 and x.size(2) == self.n_channels:
                pass
            elif x.size(1) == self.n_channels and x.size(2) == 1:
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            else:
                raise ValueError(
                    "Expected input as N,C,H,W with C=frames or N,1,D,H,W with D=frames."
                )
        else:
            raise ValueError("Expected 4D or 5D input tensor.")
        if self.ci_dim is not None:
            if ci is None:
                ci = x.new_zeros(x.size(0), x.size(2), self.ci_dim)
            elif ci.dim() == 2:
                if ci.size(1) != self.ci_dim:
                    raise ValueError("ci does not match ci_dim used to build the model.")
                ci = ci.unsqueeze(1).expand(ci.size(0), x.size(2), self.ci_dim)
            elif ci.dim() == 3:
                if ci.size(2) != self.ci_dim:
                    raise ValueError("ci does not match ci_dim used to build the model.")
                if ci.size(1) != x.size(2):
                    raise ValueError("ci temporal dimension does not match input depth.")
            else:
                raise ValueError("ci must be 2D or 3D when ci_dim is set.")
        elif ci is not None:
            raise ValueError("ci was provided but ci_dim was not set when building the model.")

        x1 = self.inc(x, ci=ci)
        x2 = self.down1(x1, ci=ci)
        x3 = self.down2(x2, ci=ci)
        x4 = self.down3(x3, ci=ci)
        x5 = self.down4(x4, ci=ci)

        x5 = self.temporal(x5)

        x = self.up1(x5, x4, ci=ci)
        x = self.up2(x, x3, ci=ci)
        x = self.up3(x, x2, ci=ci)
        x = self.up4(x, x1, ci=ci)
        x = self.outc(x)
        x = x.mean(dim=2)
        x = self.sigmoid(x)
        return x
