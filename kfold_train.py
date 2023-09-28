## PyTorch dependencies
from operator import truediv
import torch
import torch.nn as nn
# from torchvision.transforms import functional as F

## Python dependencies
import os
import json
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import pickle

## HSI-based dependencies
import spectral

## Local external libraries
from src.create_dataloaders import Get_Dataloaders
from src.models import initialize_model, translate_load_dir
from train_model import train_model, translate_net_run_dir

# Comet ML logging package
try:
    from comet_ml import Experiment
except:
    Experiment = None

#Turn off plotting
plt.ioff()
# Remove spectral warnings
spectral.settings.envi_support_nonlowercase_params = True

#! Retain information on where the directory actually is relative to the calling file
rel_call_path = os.path.dirname(__file__)

## GLOBAL Parameters to set....
COMET_PROJ_NAME = 'hyperpri'
RANDOM_STATE = 1
USE_CUDA = True
LOAD_CKPT = False

#? List out which models you want to be trained in this particular fine-tuning
model_list = {
    # 'UNET': None,
    'SpectralUNET': None,
    # 'CubeNET': None,
}

plant_metadata = {
    'train': f"{rel_call_path}/Datasets/HyperPRI/data_splits/train1.json",
    'val': f"{rel_call_path}/Datasets/HyperPRI/data_splits/val1.json",
    'test': f"{rel_call_path}/Datasets/HyperPRI/data_splits/val1.json"
}

train_params = {
    'dataset': 'HyperPRI',  # Dataset Param's -------------------
    'imgs_dir': f"{rel_call_path}/Datasets/HyperPRI",
    'json_dir': plant_metadata,   # Empty dict means to take from imgs_dir
    'num_workers': 2,
    'patch_size': (608, 968),  #(400, 600),
    'augment': False,
    'rotate': False,
    'splits': 5,
    'batch_size': {'train': 2, 'val': 2, 'test': 2},
    'num_classes': 1,
    'pretrain_dir': f"{rel_call_path}/Saved_Models/HyperPRI/",
    'hsi_lo': 25,
    'hsi_hi': 263,
    'model_name': 'SpectralUNET',  # Model Param's --------------------
    "channels": 238,  # 3 for RGB, 238 for HSI, set to 1 for 3D-UNET
    "spectral_bn_size": 64,
    "bilinear": False,
    '3d_featmaps': 128,         ## How many feature maps are in 3D_UNET's first layer
    '3d_levels': 5,            # How many 3D levels
    '3d_kernel': (1, 3, 3),    # How large the Conv3d kernel is
    '3d_poolsize': (1, 2, 2),  # How large the Pool3d kernel is
    '3d_padding': (0, 1, 1),   # Added padding for the Conv3d modules
    '3d_pooltype': 'max',      # Pick from 'max'pooling or 'avg'pooling
    'use_attention': False,
    'add_bn': False,
    'parallel_skips': False,
    'use_pretrained': False,
    'ckpt_run': 1,   # Which (split-1) to start from
    'ckpt_subdir': "SpectralUNET",
    'epochs': 1000,  # Optimizer Param's --------------------
    'lr': 0.001,
    'optim': 'adam',
    'wgt_decay': 0,
    'momentum': 0.9,
    'early_stop': 250,
    'save_epoch': 1,
    'save_cp': True  # Needs lots of disk space if you're going to do this
}

train_params['save_dir'] = f"{rel_call_path}/Saved_Models/{train_params['dataset']}/"
if LOAD_CKPT:
    load_dir = f"{train_params['save_dir']}{train_params['ckpt_subdir']}/Run_{train_params['ckpt_run']+1}/"
    max_epochs = train_params['epochs']   # Save maximum num of epochs in case we want to run the model longer
    start_run = train_params['ckpt_run']
    worker_overwrite = train_params['num_workers']
    with open(f"{load_dir}/checkpoints/ckpt_info.json", 'r') as json_file:
        saved_data = json.load(json_file)
    train_params = saved_data['params']
    train_params['epochs'] = max_epochs
    train_params['num_workers'] = worker_overwrite
    train_params['ckpt_run'] = start_run
    RANDOM_STATE = train_params['random_state']
else:
    saved_data = None
    start_run = 0
train_params['num_workers'] = 0

# Reproducibility and option for cross-validation runs (no initial seed)
if RANDOM_STATE > 0:
    train_params['random_state'] = RANDOM_STATE
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)
else:
    print(f"Initial Torch seed: {torch.seed()}")

USE_CUDA = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")  # CPU for temporary code

## For however many splits and training runs exist, evaluate the models (using eval_models)
val_metrics_list = []
test_metrics_list = []
print()
print('Starting Experiments...')
for run in range(start_run, train_params['splits']):
    # Scrappy workaround for loading a different run; models may have left off at different runs
    if LOAD_CKPT and run != start_run:  # TODO - refactor this code
        load_dir = f"{train_params['save_dir']}{train_params['ckpt_subdir']}/Run_{run+1}/"
        ckpt_filepath = f"{load_dir}/checkpoints/ckpt_info.json"
        if os.path.exists(load_dir) and os.path.exists(ckpt_filepath):
            with open(ckpt_filepath, 'r') as json_file:
                saved_data = json.load(json_file)
            train_params = saved_data['params']
        else:
            saved_data = None
        train_params['epochs'] = max_epochs

    # Change which dataset we're using for k-fold
    train_params['json_dir'] = {
        'train': f"{rel_call_path}/Datasets/HyperPRI/data_splits/train{run+1}.json",
        'val': f"{rel_call_path}/Datasets/HyperPRI/data_splits/val{run+1}.json",
        'test': f"{rel_call_path}/Datasets/HyperPRI/data_splits/val{run+1}.json"
    }

    print(train_params['json_dir'])
    ## Create dataloaders
    print("Initializing Datasets and Dataloaders...")
    all_dataloaders, pos_class_wt = Get_Dataloaders(0, train_params, train_params['batch_size'])
    train_params['pos_class_wt'] = pos_class_wt
    train_params['n_train'] = len(all_dataloaders['train'].dataset.files)
    train_params['n_val'] = len(all_dataloaders['val'].dataset.files)
    train_params['n_test'] = len(all_dataloaders['test'].dataset.files)

    print("***** TRAINING - RUN {} *****".format(run + 1))

    ## Create desired models (using initialize_model) and use pretrained if desired
    for model_key in model_list.keys():
        state_dict = None
        model_param_str = translate_load_dir(model_key, train_params)
        model_list[model_key] = initialize_model(model_key, train_params['num_classes'], train_params)
        if LOAD_CKPT:
            load_file = load_dir + f"/checkpoints/checkpoint.pth"
            state_dict = torch.load(load_file, map_location="cpu")
        elif train_params['use_pretrained']:
            load_file = train_params['pretrain_dir'] + model_param_str + f"/Run_{run+1}/best_wts.pt"
            state_dict = torch.load(load_file, map_location="cpu")
        # Having 'module' problems: not sure why saving is so wonky...
        if state_dict is not None and list(state_dict.keys())[0].startswith('module'):
            new_dict = {}
            query_str = "double_conv"
            for k in state_dict.keys():
                my_key = k.replace("module.", "", 1)
                new_dict[my_key] = state_dict[k]
            model_list[model_key].load_state_dict(new_dict)
        elif state_dict is not None:
            model_list[model_key].load_state_dict(state_dict)

    for model_key in model_list.keys():
        train_params['model_name'] = model_key

        # Comet-ML Logging initialization
        experiment = Experiment(
            api_key=None,
            project_name=COMET_PROJ_NAME,
            workspace=None,
        )
        experiment.set_name(f"{train_params['dataset']}-{model_key}-{run+1}")

        # Send the model to GPU if available
        model = model_list[model_key]
        model = model.to(device)

        # Send the model to GPU if available, use multiple if available
        if USE_CUDA and torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        
        # Print number of trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        # Train and evaluate
        logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
        logging.info(f'Using device {device}')
        logging.info(f'Network:\n'
                f'\t{train_params["channels"]} input channels\n'
                f'\t{train_params["num_classes"]} output channels (classes)\n'
                f'\t{"Bilinear" if train_params["bilinear"] else "Transposed conv"} upscaling\n'
                f'\tTotal number of trainable parameters: {num_params}')

        print('Starting Experiments...')
        experiment.log_parameters(train_params)
        experiment.set_model_graph(model_list[model_key])

        #? Train and evaluate the model
        train_model(model_list[model_key],
                    dataloaders=all_dataloaders,
                    train_params=train_params,
                    device=device,
                    split=run,
                    save_dir=train_params['save_dir'],
                    save_cp=train_params['save_cp'],
                    save_epoch=train_params['save_epoch'],
                    comet_exp=experiment,
                    ckpt_data=saved_data)
