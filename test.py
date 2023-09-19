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
from src.prediction_mask import eval_models
from src.save_spreadsheet import fill_metrics_spreadsheet

## Parameters to set....
rel_call_path = os.path.dirname(__file__)

## Parameters to set....
random_state = 1
use_cuda = True

#? List out which models you want to be trained in this particular fine-tuning
model_list = {
    # 'XuNET': None,
    'UNET': None,
    # 'JOSHUA': None,
    # 'JOSHUAres': None,
    # 'BNH': None,
    # '3D_UNET': None,
    # '3D_BNH': None
}

plant_metadata = {
    'train': f"{rel_call_path}/Datasets/HyperPRI/phenotype2.json",
    'val': f"{rel_call_path}/Datasets/HyperPRI/phenotype1.json",
    'test': f"{rel_call_path}/Datasets/HyperPRI/phenotype1.json"
}

test_params = {
    'dataset': 'HyperPRI',  # Dataset Param's
    'imgs_dir': f"{rel_call_path}/Datasets/HyperPRI",
    'json_dir': plant_metadata,
    'num_workers': 2,
    'patch_size': (608, 968),
    'rescale': 1,
    'augment': False,
    'rotate': False,
    'splits': 2,
    'batch_size': {'train': 1, 'val': 1, 'test': 1},
    'num_classes': 1,
    'pretrain_dir': f"{rel_call_path}/Saved_Models/HyperPRI/",
    'hsi_lo': 0,
    'hsi_hi': 0,
    'model_name': 'UNET',  # Model Param's
    "channels": 3,
    "bilinear": False,
    "feature_extraction": False,
    "use_attention": False,
    "histogram_skips": False,  # T - JOSHUA, F - BNH
    "histogram_pools": True,   # T - BNH, F - JOSHUA
    'numBins': 4,
    'normalize_count': True,
    'normalize_bins': True,
    'skip_locations': [True, True, True, True],
    'pool_locations': [False, False, False, True],  # BNH is [F, F, F, T]
    'hist_size': [2, 2, 2, 2],
    '3d_featmaps': 8,          # How many feature maps are in 3D_UNET's first layer
    '3d_levels': 5,            # How many 3D levels
    '3d_kernel': (9, 3, 3),    # How large the Conv3d kernel is
    '3d_poolsize': (1, 2, 2),  # How large the Pool3d kernel is
    '3d_padding': (4, 1, 1),   # Added padding for the Conv3d modules
    '3d_pooltype': 'max',      # Pick from 'max'pooling or 'avg'pooling
    'use_attention': False,
    'add_bn': False,
    'parallel_skips': False,
    'use_pretrained': True,
    'hist_reg': 0.0,
    'epochs': 30,  # Optimizer Param's
    'lr': 1e-5,
    'optim': 'sgd',
    'wgt_decay': 1e-7,
    'momentum': 0.9,
    'early_stop': 5,
    'save_maps': False,  # File-specific Param's
    'only_validation': False,
    'thresholds': [
        [
            # 0.32,
            # 0.36,
            # 0.42,
            # 0.5,
            0.54,    # No-pretraining UNET 0.72 - BN, 0.60 - GN
            # 0.58,    # No-pretraining BNH
            # 0.02,  #0.36,  # HSI-trained UNET
            # 0.12,  #0.24,  # HSI-trained BNH
            # 0.5,     # 3D-UNET on 224 channels
        ],
        [
            0.56,  #0.72 - BN, 0.78 - GN
            # 0.62,
        ],
        [
            0.12,  # HSI-trained UNET
            # 0.02,  # HSI-trained BNH
        ]
    ],
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # CPU for temporary code

## Create dataloaders
all_dataloaders, _ = Get_Dataloaders(0, test_params, test_params['batch_size'])

## For however many splits and training runs exist, evaluate the models (using eval_models)
val_metrics_list = []
test_metrics_list = []
for run in range(0, test_params['splits']):
    print("***** TESTING - RUN {} *****".format(run + 1))
    ## Create desired models (using initialize_model) and use pretrained if desired
    for model_key in model_list.keys():
        model_param_str = translate_load_dir(model_key, test_params)
        model_list[model_key] = initialize_model(model_key, test_params['num_classes'], test_params)
        if model_param_str == 'PrmiNET':
            model_param_str = "XuNET"
        load_file = test_params['pretrain_dir'] + model_param_str + f"/Run_{run+1}/best_wts.pt"
        state_dict = torch.load(load_file, map_location="cpu")

        if list(state_dict.keys())[0].startswith('module'):
            new_dict = {}
            query_str = "double_conv"
            for k in state_dict.keys():
                my_key = k.replace("module.", "", 1)
                new_dict[my_key] = state_dict[k]
            model_list[model_key].load_state_dict(new_dict)
        else:
            model_list[model_key].load_state_dict(state_dict)

    ## Generate and save prediction masks
    saved_metrics = eval_models(model_list,
                                dataloaders=all_dataloaders,
                                mask_type=bool,
                                device=device,
                                split=run,
                                params=test_params,
                                thresholds=test_params['thresholds'][run],
                                save_dir=f"{rel_call_path}/Saved_Models/{test_params['dataset']}_finetune/",
                                save_mask_png=test_params['save_maps'],
                                only_val=test_params['only_validation'])

    val_metrics_list.append(saved_metrics[0])
    if not test_params['only_validation']:
        test_metrics_list.append(saved_metrics[1])

    with open(test_params['pretrain_dir'] + f"/val_{run + 1}.pkl", 'wb') as temp_file:
        temp_metrics = pickle.dump(saved_metrics[0], temp_file)

    if not test_params['only_validation']:
        with open(test_params['pretrain_dir'] + f"/test_{run + 1}.pkl", 'wb') as temp_file:
            temp_metrics = pickle.dump(saved_metrics[1], temp_file)

## Save and Print out statistics at the end of each run,
##    One for validation and one for test
fill_metrics_spreadsheet(val_metrics_list, model_list.keys(),
                         test_params['pretrain_dir'] + "/",
                         file_name="val_hyperpri_metrics")
if not test_params['only_validation']:
    fill_metrics_spreadsheet(test_metrics_list, list(model_list.keys()),
                            test_params['pretrain_dir'] + "/",
                            file_name="test_hyperpri_metrics")
