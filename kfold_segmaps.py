'''
kfold_segmaps.py

File Purpose: Acquire testing metrics using the best
    thresholds per each model; output segmentation masks.

@author: changspencer
'''
## PyTorch dependencies
import torch

## Python dependencies
import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import pickle

## HSI-based dependencies
import spectral

## PyTorch Lightning Imports
import lightning.pytorch as pl

## Local external libraries
from src.Experiments.params_HyperPRI import ExpRedGreenBluePRI, ExpHyperspectralPRI
from src.PLTrainer import test_net

# Remove spectral warnings
spectral.settings.envi_support_nonlowercase_params = True

#! Retain information on where the directory actually is relative to the calling file
rel_call_path = os.path.dirname(os.path.abspath(__file__))

## GLOBAL Parameters to set....
RANDOM_STATE = 1
USE_CUDA = True
LOAD_CKPT = False
TEST_AUG = False

start_split = 0
num_splits = 5   # Assuming multiple splits

update_params = {
    'model_name': [
        # 'UNET',
        'SpectralUNET',
        'CubeNET',
    ],
    'dataset': [
        # 'RGB',
        'HSI',
        'HSI',
    ],
    'criterion': [
        torch.nn.BCEWithLogitsLoss(),
        torch.nn.BCEWithLogitsLoss(),
        torch.nn.BCEWithLogitsLoss(),  # Need to handle previous changes to the code
    ]
}
thresholds = [
    # [0.36, 0.37, 0.38, 0.39, 0.45],   # UNET
    [0.30, 0.40, 0.43, 0.37, 0.34],   # SpectralUNET
    [0.38, 0.41, 0.47, 0.48, 0.42],   # CubeNET
]
segmaps = [
    #True,
    True,
    True,
    # False,
    # False
]
models = update_params['model_name']
datasets = update_params['dataset']
testing_set = 'val'

plt_colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
]

## File-specific parameters to set....
random_state = 1
use_cuda = True
print("\n ~~~~~~~~~~ 5-SPLIT CYCLES ~~~~~~~~~~\n")
plt.figure(dpi=150)
for run in range(start_split, num_splits):
    print(f" ********** Split {run+1} **********")

    for m_idx, (m, dset) in enumerate(zip(models, datasets)):
        change_params = {}
        for k_idx, k in enumerate(update_params):
            change_params[k] = update_params[k][m_idx]

        if dset.lower() == 'rgb':
            exp_params = ExpRedGreenBluePRI(rel_call_path, split_no=run+1, augment=TEST_AUG)
            # Switch to a different model (ie. change internal parameter strings)
            exp_params.change_network_param(m, rel_call_path, run+1, model_params=change_params)  # Num bins
        else:
            exp_params = ExpHyperspectralPRI(rel_call_path, split_no=run+1)
            # Switch to a different model (ie. change internal parameter strings)
            exp_params.change_network_param(m, rel_call_path, run+1)  # Num bins

        print(f"   Model: {exp_params.model_param_str}")
        print(f"   Test JSON: {exp_params.json_dir[testing_set]}")
        if testing_set == 'train':
            data_obj = exp_params.get_train_data()
        if testing_set == 'val':
            data_obj = exp_params.get_val_data()
        if testing_set == 'test':
            data_obj = exp_params.get_test_data()

        pr_curve_info = test_net(data_obj,
                                 exp_params,
                                 best_threshold=thresholds[m_idx][run],
                                 save_segmaps=segmaps[m_idx])

        if run == start_split:
            label_str = f"{exp_params.model_name}"
        else:
            label_str = None
