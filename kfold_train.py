## Python dependencies
import os
import datetime

# import numpy as np
import matplotlib.pyplot as plt

## PyTorch Lightning Imports
# import torch.distributed as torch_dist
# import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_only

## HSI-based dependencies
import spectral

## Local external libraries
from src.Experiments.params_HyperPRI import ExpRedGreenBluePRI, ExpHyperspectralPRI
from src.PLTrainer import train_net, validate_net

# Comet ML logging package
try:
    from comet_ml import Experiment
    online_comet = True
except:
    online_comet = False

#Turn off plotting
plt.ioff()
# Remove spectral warnings
spectral.settings.envi_support_nonlowercase_params = True

@rank_zero_only
def rename_folder(save_path):
    # Move the old set of directories if the `save_path` is taken...
    now_time = datetime.datetime.now()
    time_str = "_{}{}{}_{}{}{}".format(now_time.year,
                                    now_time.month,
                                    now_time.day,
                                    now_time.hour,
                                    now_time.minute,
                                    now_time.second)
    if os.path.exists(save_path):
        print("\n!!!!!! ----- RENAMING FOLDER ----- !!!!!!\n")
        # Assumes that the last character is '/' and that there's only one there.
        os.rename(save_path, save_path[:-1] + time_str + '/')


if __name__ == "__main__":
    #! Retain information on where the directory actually is relative to the calling file
    rel_call_path = os.path.dirname(os.path.abspath(__file__))

    ## GLOBAL Parameters to set....
    RANDOM_STATE = 1
    USE_CUDA = True
    MODEL_SHARD = False   # SpectralUNET: Use DeepZeRO-2 to parallelize model
    LOAD_CKPT = False  # Assumes that the `start_split` index will load while subsequent will not.
    DATA_AUG = False

    n_seeds = 1
    start_split = 0
    num_splits = 5   # Assuming multiple splits
    dataset = "HSI"

    print("\n ~~~~~~~~~~ 5-SPLIT CYCLES ~~~~~~~~~~\n")
    if MODEL_SHARD:
        for k in os.environ.keys():
            if "rank" in k.lower() or "member" in k.lower() or "node" in k.lower():
                print(f"{k} - {os.environ[k]}")

    for run in range(start_split, num_splits):
        print(f" ********** Split {run+1} **********")

        for seed_idx in range(n_seeds):
            print(f"        Seed {seed_idx+1} / {n_seeds}.....")
            os.system("nvidia-smi")

            if dataset.lower() == 'rgb':
                exp_params = ExpRedGreenBluePRI(rel_call_path, split_no=run+1, seed_num=seed_idx,
                                                augment=DATA_AUG, comet_logging=online_comet)
            else:
                exp_params = ExpHyperspectralPRI(rel_call_path, split_no=run+1, seed_num=seed_idx,
                                                 comet_logging=online_comet)

            # rename_folder(exp_params.save_path)
            pl_trainer = train_net(exp_params, checkpoint=LOAD_CKPT, model_parallel=MODEL_SHARD)
            if n_seeds > 1:
                print(f"   Model: {exp_params.model_param_str}")
                print(f"   Validation JSON: {exp_params.json_dir['val']}")
                validate_net(exp_params.get_val_data(),
                             exp_params,
                             save_segmaps=False)
        LOAD_CKPT = False
