'''
validate.py

File Purpose: Evaluate the best threshold value for fine-tuned
    models using the existing validation dataset

@author: changspencer
'''
## PyTorch dependencies
import torch
# import torch.nn as nn
# from torchvision.transforms import functional as F

## Python dependencies
import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import pickle

## Local external libraries
from src.create_dataloaders import Get_Dataloaders
from src.models import initialize_model, translate_load_dir
from src.prediction_mask import eval_models, model_pr_stats
from src.save_spreadsheet import fill_metrics_spreadsheet

## Parameters to set....

#! Retain information on where the directory actually is
rel_call_path = os.path.dirname(__file__)

## File-specific parameters to set....
random_state = 1
use_cuda = True

#? List out which models you want to be trained in this particular fine-tuning
model_list = {
    'UNET': None,
    'SpectralUNET': None,
    'CubeNET': None,
}

plant_metadata = {
    'train': f"{rel_call_path}/Datasets/HyperPRI/JSON_splits/train1.json",
    'val': f"{rel_call_path}/Datasets/HyperPRI/JSON_splits/val1.json",
    'test': f"{rel_call_path}/Datasets/HyperPRI/JSON_splits/val1.json"
}

validate_params = {
    'dataset': 'HyperPRI',  # Dataset Param's
    'imgs_dir': f"{rel_call_path}/Datasets/HyperPRI",
    'json_dir': plant_metadata,
    'num_workers': 1,
    'patch_size': (608, 968),
    'augment': False,
    'rotate': False,
    'splits': 1,
    'batch_size': {'train': 1, 'val': 1, 'test': 1},
    'num_classes': 1,
    'pretrain_dir': f"{rel_call_path}/Saved_Models/HyperPRI/",
    'hsi_lo': 25,
    'hsi_hi': 263,
    "spectral_bn_size": 16,
    'model_name': 'UNET',  # Model Param's
    "channels": 3,
    "bilinear": False,
    "feature_extraction": False,
    "use_attention": False,
    'skip_locations': [True, True, True, True],
    'pool_locations': [False, False, False, True],  # BNH is [F, F, F, T]
    'hist_size': [2, 2, 2, 2],
    '3d_featmaps': 64,          # How many feature maps are in 3D_UNET's first layer
    '3d_levels': 5,            # How many 3D levels
    '3d_kernel': (9, 3, 3),    # How large the Conv3d kernel is
    '3d_poolsize': (1, 2, 2),  # How large the Pool3d kernel is
    '3d_padding': (4, 1, 1),   # Added padding for the Conv3d modules
    '3d_pooltype': 'avg',      # Pick from 'max'pooling or 'avg'pooling
    'add_bn': False,
    'parallel_skips': False,
    'use_pretrained': True,
    'epochs': 30,  # Optimizer Param's
    'lr': 1e-5,
    'optim': 'adam',
    'wgt_decay': 1e-7,
    'momentum': 0.9,
    'early_stop': 5,
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # CPU for temporary code

model_pr_stats(model_list,
               rel_call_path=rel_call_path,
               mask_type=bool,
               device=device,
               params=validate_params,
               save_dir=f"{rel_call_path}/Saved_Models/{validate_params['dataset']}/")