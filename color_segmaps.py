# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
Note: Script should be used if ALL models are saved out
If only interested in certain models, modify the "seg_models" (Line 69)
dictionary to only include models of interests
@author: jpeeples
"""
## PyTorch dependencies
from operator import truediv
import torch
import torch.nn as nn
# from torchvision.transforms import functional as F

## Python dependencies
import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import pickleshare

## Local external libraries
from src.create_dataloaders import Get_Dataloaders
from src.Create_Individual_RGB_Figures import Generate_Images

#Turn off plotting
plt.ioff()

#! Retain information on where the directory actually is
rel_call_path = os.path.dirname(__file__)

## Parameters to set....
comet_proj_name = 'hyperpri'
random_state = 1
use_cuda = True

#? List out which models you want to be trained in this particular fine-tuning
model_list = [
    # 'XuNET',
    'UNET',
    # 'JOSHUA',
    # 'JOSHUAres',
    'BNH'
]
thresholds = [
    # 0.42,
    # 0.4,
    # 0.42,
    # 0.5
    0.36,  # HSI-UNET
    0.24,  # HSI-BNH
]

vis_params = {
    'dataset': 'HyperPRI',  # Dataset Param's
    'imgs_dir': f"{rel_call_path}/Datasets/HyperPRI",
    'num_workers': 1,
    'patch_size': (608, 968),
    'augment': False,
    'rotate': False,
    'splits': 1,
    'batch_size': {'train': 4, 'val': 2, 'test': 2},
    'num_classes': 1,
    'pretrain_dir': f"{rel_call_path}/Saved_Models/HyperPRI/",
    'model_name': 'UNET',  # Model Param's
    "channels": 299,
    "bilinear": False,
    "feature_extraction": False,
    "use_attention": False,
    # '3d_featmaps': 8,          # How many feature maps are in 3D_UNET's first layer
    # '3d_levels': 5,            # How many 3D levels
    # '3d_kernel': (7, 3, 3),    # How large the Conv3d kernel is
    # '3d_poolsize': (1, 2, 2),  # How large the Pool3d kernel is
    # '3d_padding': (3, 1, 1),   # Added padding for the Conv3d modules
    # '3d_pooltype': 'max',      # Pick from 'max'pooling or 'avg'pooling
    'use_attention': False,
    'add_bn': False,
    'parallel_skips': False,
    'use_pretrained': True,
    'epochs': 200,  # Optimizer Param's
    'lr': 5e-4,
    'optim': 'sgd',
    'wgt_decay': 1e-7,
    'momentum': 0.9,
    'early_stop': 30,
    'save_epoch': 99,
    'save_cp': True  # Needs lots of disk space if you're going to do this
}
vis_params['thresholds'] = thresholds

for key in list(vis_params.keys()):
    print(f"{key} - {vis_params[key]}")

# Reproducibility and option for cross-validation runs (no initial seed)
if random_state > 0:
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
else:
    print(f"Initial Torch seed: {torch.seed()}")

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")  # CPU for temporary code

## Create dataloaders
print("Initializing Datasets and Dataloaders...")
all_dataloaders, pos_class_wt = Get_Dataloaders(0, vis_params, vis_params['batch_size'])
vis_params['pos_class_wt'] = pos_class_wt
vis_params['n_train'] = len(all_dataloaders['train'].dataset.files)
vis_params['n_val'] = len(all_dataloaders['val'].dataset.files)
vis_params['n_test'] = len(all_dataloaders['test'].dataset.files)

metrics = {'dice': 'Dice Coefficent', 'overall_IOU': 'IOU','pos_IOU': 'Positive IOU',
            'haus_dist': 'Hausdorff distance', 'adj_rand': 'Adjusted Rand Index',
            'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1 Score',
            'specificity': 'Specificity',
            'pixel_acc': 'Pixel Accuracy','loss': 'Binary Cross Entropy',
            'inf_time': 'Inference Time'}
# seg_models = {0: 'UNET', 1: 'UNET+', 2: 'Attention_UNET', 3:'JOSHUA', 4: 'JOSHUA+'}
# seg_models = {3: 'JOSHUA', 4: 'JOSHUA+', 5: 'JOSHUAres'}

#Return datasets and indices of training/validation data
dataloaders, pos_wt = Get_Dataloaders(vis_params, vis_params, vis_params['batch_size'])
# print(vis_params)

mask_type = torch.float32  # binary segmentation

for split in range(0, vis_params['splits']):
    #Save figures for individual images
    Generate_Images(dataloaders, mask_type, model_list, device,split,
                    vis_params['num_classes'], vis_params, alpha=.6,
                    use_cuda=use_cuda)

    print('**********Run ' + str(split+1) + ' Finished**********')
