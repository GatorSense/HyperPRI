'''
kfold_validate.py

File Purpose: Evaluate the best threshold value for fine-tuned
    models using the existing validation dataset

@author: changspencer
'''
## PyTorch dependencies
import torch

## Python dependencies
import os
import matplotlib.pyplot as plt

## HSI-based dependencies - Remove spectral warnings
import spectral
spectral.settings.envi_support_nonlowercase_params = True

## Local external libraries
from src.Experiments.params_HyperPRI import ExpRedGreenBluePRI, ExpHyperspectralPRI
from src.PLTrainer import validate_net

#! Retain information on where the directory actually is relative to the calling file
rel_call_path = os.path.dirname(os.path.abspath(__file__))

## GLOBAL Parameters to set....
RANDOM_STATE = 1
USE_CUDA = True
LOAD_CKPT = False
TEST_AUG = False

n_seeds = 1
start_split = 0
num_splits = 5   # Assuming multiple splits

update_params = {
    'model_name': [
         'UNET',
         # 'SpectralUNET',
        #  'CubeNET',
    ],
    'dataset': [
        'RGB',
        # 'HSI',
        # 'HSI',
    ],
    'spectral_bn_size': [
        0,
        1650,
        0,
    ],
    'cube_featmaps': [
        0,
        0,
        64,
    ],
    'criterion': [
        torch.nn.BCEWithLogitsLoss(),
        torch.nn.BCEWithLogitsLoss(),
        torch.nn.BCEWithLogitsLoss(),
    ]
}
models = update_params['model_name']
datasets = update_params['dataset']
segmaps = [
    False,
    False,
    False,
    # True,
    # True,
    # True,
]

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

        for seed_idx in range(n_seeds):  # In case of running multiple random seeds on one split
            split_no = seed_idx * 10 + run + 1
            if dset.lower() == 'rgb':
                exp_params = ExpRedGreenBluePRI(rel_call_path, split_no=run+1, augment=TEST_AUG)
                # Switch to a different model (ie. change internal parameter strings)
                exp_params.change_network_param(m, rel_call_path, run+1, model_params=change_params)  # Num bins
            else:
                exp_params = ExpHyperspectralPRI(rel_call_path, split_no=run+1)
                # Switch to a different model (ie. change internal parameter strings)
                exp_params.change_network_param(m, rel_call_path, run+1, model_params=None)  # Num bins


            print(f"   Model: {exp_params.model_param_str}")
            print(f"   Validation JSON: {exp_params.json_dir['val']}")
            # pr_curve_info = validate_net(exp_params.get_val_data(),
            pr_curve_info = validate_net(exp_params.get_val_data(),
                                         exp_params,
                                         save_segmaps=segmaps[m_idx])

        # Will plot the last of all potential seeded runs
        if run == start_split:
            label_str = f"{exp_params.model_name}"
        else:
            label_str = None
        plt.plot(pr_curve_info[1], pr_curve_info[0], alpha=0.7,
                 color=plt_colors[m_idx], label=label_str)

curve_str = "_".join(models)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.legend()

plt.savefig(f"{rel_call_path}/Saved_Models/{dset}_finetune/{curve_str}_pr.png")
plt.show()
