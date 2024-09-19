"""
File composed of all learning models' class declarations and parts/modules.

Takes after previous code by jpeeples67 in https://github.com/GatorSense/Histological_Segmentation

@author: changspencer
"""

## PyTorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,
                 feature_extraction = False, use_attention = False,
                 analyze=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention
        self.analyze = analyze
        factor = 2 if bilinear else 1

        starter_dim = 64
        out1 = starter_dim * 2
        out2 = starter_dim * 2**2
        out3 = starter_dim * 2**3
        out4 = starter_dim * 2**4

        self.inc = DoubleConv(n_channels, starter_dim)
        self.down1 = Down(starter_dim, out1)
        self.down2 = Down(out1, out2)
        self.down3 = Down(out2, out3)
        self.down4 = Down(out3, out4 // factor)

        self.up1 = Up(out4, out3, bilinear, use_attention=self.use_attention)
        self.up2 = Up(out3, out2, bilinear, use_attention=self.use_attention)
        self.up3 = Up(out2, out1, bilinear, use_attention=self.use_attention)
        self.up4 = Up(out1, starter_dim * factor, bilinear, use_attention=self.use_attention)
        self.outc = OutConv(starter_dim, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if self.analyze:
            return (logits, logits, torch.sigmoid(logits))
        else:
            return logits


class SpectralUNET(torch.nn.Module):
    def __init__(self, hsi_depth, n_classes, bn_feats=16, bnorm=True):
        """
        MLP Autoencoder architecture that is UNET-like.
            Its current implementation utilizes a single learned layer size
            across all encoder-decoder layers.
        """
        super(SpectralUNET, self).__init__()

        self.hsi_depth = hsi_depth
        self.n_channels = hsi_depth
        self.n_classes = n_classes

        # May be changed if user desires different feature numbers in each layer
        self.layer_feats = [
            bn_feats,
            bn_feats,
            bn_feats,
            bn_feats,
            bn_feats,
        ]

        self.tail = self._basic_module(hsi_depth, self.layer_feats[-1], bn=bnorm)
        self.down1 = self._basic_module(self.layer_feats[-1], self.layer_feats[-2], bn=bnorm)
        self.down2 = self._basic_module(self.layer_feats[-2], self.layer_feats[-3], bn=bnorm)
        self.down3 = self._basic_module(self.layer_feats[-3], self.layer_feats[-4], bn=bnorm)
        self.down4 = self._basic_module(self.layer_feats[-4], bn_feats, bn=bnorm)
        self.up1 = self._basic_module(bn_feats, self.layer_feats[-4], bn=bnorm)   # starts with down4
        self.up2 = self._basic_module(2*self.layer_feats[-3], self.layer_feats[-3], bn=bnorm)  # concat's with down3
        self.up3 = self._basic_module(2*self.layer_feats[-2], self.layer_feats[-2], bn=bnorm)  # concat's with down2
        self.up4 = self._basic_module(2*self.layer_feats[-1], self.layer_feats[-1], bn=bnorm)  # concat's with down1
        self.outc = torch.nn.Linear(2*bn_feats,
                                    self.n_classes) # concat's with "flattened" tail

    def _basic_module(self, in_feats, out_feats, bn=True):
        if not bn:
            return torch.nn.Sequential(
            torch.nn.Linear(in_feats, out_feats),
            torch.nn.ReLU()
            )
        return torch.nn.Sequential(
            torch.nn.Linear(in_feats, out_feats),
            torch.nn.BatchNorm1d(out_feats),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        """
        Expected input x is N x D x R x C
            N - number of images
            D - depth of image (matches self.hsi_depth)
            R - num of image rows
            C - num of image cols
        """
        if not x.shape[1] == self.hsi_depth:
            ValueError(x.shape[1])
        x_row = x.shape[-2]
        x_col = x.shape[-1]

        rast_x = x.reshape(x.shape[0], x.shape[1], x.shape[-2] * x.shape[-1]).permute((0, 2, 1))
        out_x = torch.zeros((x.shape[0], self.n_classes, x_row, x_col), device=x.device)
        for idx, in_x in enumerate(rast_x):
            x0 = self.tail(in_x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            x4 = self.down4(x3)

            tmp_x = self.up1(x4)
            tmp_x = self.up2(torch.cat((x3, tmp_x), axis=-1))
            tmp_x = self.up3(torch.cat((x2, tmp_x), axis=-1))
            tmp_x = self.up4(torch.cat((x1, tmp_x), axis=-1))
            tmp_x = self.outc(torch.cat((x0, tmp_x), axis=-1))
            out_x[idx] = tmp_x.reshape(self.n_classes, x_row, x_col)
        return out_x


class CubeNET(torch.nn.Module):
    def __init__(self, hsi_depth, n_classes, first_depth=64,
                 bilinear=True, use_attention=False,
                 analyze=False):
        """
        A UNET that begins first with a 3D convolutional layer.
            After the first 3D convolutional layer to grab all
            hyperspectral bands, the architecture is the same as the UNET.
        """
        super(CubeNET, self).__init__()

        self.n_channels = 1
        self.depth = hsi_depth
        self.first_depth = first_depth
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention
        self.analyze = analyze
        factor = 2 if bilinear else 1

        # Completely quantizes the initial HSI data with a weighted mean.
        self.first_conv = torch.nn.Conv3d(1, first_depth, kernel_size=(self.depth, 3, 3), padding=(0, 1, 1))
        self.inc = torch.nn.Sequential(
            self.first_conv,
            torch.nn.BatchNorm3d(first_depth),
            torch.nn.ReLU(inplace=True),
        )
        # Need to match UNET's starter "DoubleConv" module.
        self.inc2 = torch.nn.Sequential(
            torch.nn.Conv2d(first_depth, first_depth, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(first_depth),
            torch.nn.ReLU(inplace=True),
        )
        C = 128

        self.down1 = Down(first_depth, C)
        self.down2 = Down(C, C * 2)
        self.down3 = Down(C * 2, C * 2**2)
        self.down4 = Down(C * 2**2, C * 2**3 // factor)

        self.up1 = Up(C * 2**3, C * 2**2, self.bilinear, use_attention=self.use_attention)
        self.up2 = Up(C * 2**2, C * 2, self.bilinear, use_attention=self.use_attention)
        self.up3 = Up(C * 2, C, self.bilinear, use_attention=self.use_attention)
        if first_depth == 64:
            self.up4 = Up(C, 64 * factor, self.bilinear, use_attention=self.use_attention)
        else:
            if self.bilinear:
                self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.upconv4 = DoubleConv(C + first_depth, 64, 64)
            else:
                self.upsample4 = nn.ConvTranspose2d(C, 64, kernel_size=2, stride=2)
                self.upconv4 = DoubleConv(64 + first_depth, 64)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        """
        Expected input x is N x 1 x D x R x C
            N - number of images
            D - depth of image (matches self.hsi_depth)
            R - num of image rows
            C - num of image cols
        """
        if not x.shape[2] == self.depth:
            ValueError(x.shape[2])
        x_row = x.shape[-2]
        x_col = x.shape[-1]

        x1 = self.inc(x)
        x1 = x1.reshape((x.shape[0], self.first_conv.out_channels, x_row, x_col))
        x1 = self.inc2(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        # Working with two CubeNET versions
        if self.first_depth == 64:
            x = self.up4(x, x1)
        else:
            x = self.upsample4(x)
            # input is CHW
            diffY = torch.tensor([x1.size()[2] - x.size()[2]])
            diffX = torch.tensor([x1.size()[3] - x.size()[3]])

            x = F.pad(x, [torch.div(diffX, 2, rounding_mode='floor'),
                            diffX - torch.div(diffX, 2, rounding_mode='floor'),
                            torch.div(diffY, 2, rounding_mode='floor'),
                            diffY - torch.div(diffY, 2, rounding_mode='floor')])
            x = torch.cat([x1, x], dim=1)
            x = self.upconv4(x)

        logits = self.outc(x)

        if self.analyze:
            return (logits, logits, torch.sigmoid(logits))
        else:
            return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=False):
        super().__init__()

        self.use_attention = use_attention
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if self.use_attention:
                self.conv = DoubleConv(in_channels // 2, out_channels // 2, in_channels // 2)
            else:
                self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2,
                                             kernel_size=2, stride=2)
            if self.use_attention:
                self.conv = DoubleConv(in_channels // 2, out_channels)
            else:
                self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='floor'),
                        diffX - torch.div(diffX, 2, rounding_mode='floor'),
                        torch.div(diffY, 2, rounding_mode='floor'),
                        diffY - torch.div(diffY, 2, rounding_mode='floor')])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.use_attention:
            x = x2*x1
        else:
            x = torch.cat([x2, x1], dim=1)


        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)