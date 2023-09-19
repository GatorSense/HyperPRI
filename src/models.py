"""
File composed of all learning models' class declarations.

@author: ?
"""
import numpy as np

## PyTorch dependencies
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

from .model_parts import *


def set_parameter_requires_grad(model, feature_extraction):
    if feature_extraction:
        for param in model.parameters():
            param.requires_grad = False


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

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
            
        self.up1 = Up(1024, 512, bilinear, use_attention=self.use_attention)
        self.up2 = Up(512, 256, bilinear, use_attention=self.use_attention)
        self.up3 = Up(256, 128, bilinear, use_attention=self.use_attention)
        self.up4 = Up(128, 64 * factor, bilinear, use_attention=self.use_attention)
        self.outc = OutConv(64, n_classes)

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
    def __init__(self, hsi_depth, n_classes, bn_feats=16):
        super(SpectralUNET, self).__init__()

        self.hsi_depth = hsi_depth
        self.n_channels = hsi_depth
        self.n_classes = n_classes

        self.layer_feats = [
            bn_feats,
            bn_feats, # * 2,
            bn_feats, #  * 4,
            bn_feats, # * 8,
            bn_feats, # * 16,
        ]

        self.tail = self._basic_module(hsi_depth, self.layer_feats[-1])
        self.down1 = self._basic_module(self.layer_feats[-1], self.layer_feats[-2])
        self.down2 = self._basic_module(self.layer_feats[-2], self.layer_feats[-3])
        self.down3 = self._basic_module(self.layer_feats[-3], self.layer_feats[-4])
        self.down4 = self._basic_module(self.layer_feats[-4], bn_feats)
        self.up1 = self._basic_module(bn_feats, self.layer_feats[-4])   # starts with down4
        self.up2 = self._basic_module(2*self.layer_feats[-3], self.layer_feats[-3])  # concat's with down3
        self.up3 = self._basic_module(2*self.layer_feats[-2], self.layer_feats[-2])  # concat's with down2
        self.up4 = self._basic_module(2*self.layer_feats[-1], self.layer_feats[-1])  # concat's with down1
        self.outc = torch.nn.Linear(2*bn_feats, # * 32,
                                    self.n_classes) # concat's with "flattened" tail

    def _basic_module(self, in_feats, out_feats, bn=True):
        if not bn:
            return torch.nn.Sequential(
            torch.nn.Linear(in_feats, out_feats),
            torch.nn.ReLU()
            )
        return torch.nn.Sequential(
            torch.nn.Linear(in_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(out_feats)
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
            # del x4
            tmp_x = self.up2(torch.cat((x3, tmp_x), axis=-1))
            # del x3
            tmp_x = self.up3(torch.cat((x2, tmp_x), axis=-1))
            # del x2
            tmp_x = self.up4(torch.cat((x1, tmp_x), axis=-1))
            # del x1
            tmp_x = self.outc(torch.cat((x0, tmp_x), axis=-1))
            # del x0
            # tmp_x = self.up1(x4)
            # tmp_x = torch.cat((x3, tmp_x), axis=-1)
            # tmp_x = self.up2(tmp_x)
            # tmp_x = torch.cat((x2, tmp_x), axis=-1)
            # tmp_x = self.up3(tmp_x)
            # tmp_x = torch.cat((x1, tmp_x), axis=-1)
            # tmp_x = self.up4(tmp_x)
            # tmp_x = torch.cat((x0, tmp_x), axis=-1)
            # tmp_x = self.outc(tmp_x)
            out_x[idx] = tmp_x.reshape(self.n_classes, x_row, x_col)
        return out_x


class CubeNET(torch.nn.Module):
    def __init__(self, hsi_depth, n_classes, first_depth=64,
                 bilinear=True, use_attention=False,
                 analyze=False):
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
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm3d(first_depth)
        )
        self.down1 = Down(first_depth, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512, bilinear, use_attention=self.use_attention)
        self.up2 = Up(512, 256, bilinear, use_attention=self.use_attention)
        self.up3 = Up(256, 128, bilinear, use_attention=self.use_attention)
        if first_depth == 64:
            self.up4 = Up(128, 64 * factor, bilinear, use_attention=self.use_attention)
        else:
            if bilinear:
                self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.upconv4 = DoubleConv(128 + first_depth, 64, 64)
            else:
                self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.upconv4 = DoubleConv(64 + first_depth, 64)
        self.outc = OutConv(64, n_classes)

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




def initialize_model(model_name, num_classes, Network_parameters, analyze=False):
    """
    Initializes model based on given model name string and provided parameters
    """
    #Base UNET model or UNET+ (our version of attention)
    if model_name == 'UNET': 
        model = UNet(Network_parameters['channels'], num_classes,
                     bilinear = Network_parameters['bilinear'],
                     feature_extraction = Network_parameters['feature_extraction'],
                     use_attention=Network_parameters['use_attention'],
                     analyze=analyze)
        
    elif model_name == 'SpectralUNET': 
        depth = Network_parameters['hsi_hi'] - Network_parameters['hsi_lo']
        model = SpectralUNET(depth, num_classes, bn_feats=Network_parameters['spectral_bn_size'])

    elif model_name == 'CubeNET':
        depth = Network_parameters['hsi_hi'] - Network_parameters['hsi_lo']
        model = CubeNET(depth, num_classes, first_depth=Network_parameters['3d_featmaps'],
                        bilinear = Network_parameters['bilinear'],
                        use_attention=Network_parameters['use_attention'],
                        analyze=analyze)

    else: #Show error that segmentation model is not available
        raise RuntimeError('Invalid model')


    return model


def translate_load_dir(model_name, net_params):    #Generate segmentation model
    """
    Translate the 
    """
    if model_name == 'SpectralUNET':
        model_str = f"{model_name}_{net_params['spectral_bn_size']}"
    elif model_name == 'CubeNET':
        model_str = f"{model_name}_{net_params['3d_featmaps']}"
    #Base UNET model
    else:
        model_str = "UNET"
    
    return model_str
