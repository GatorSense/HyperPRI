# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:46:58 2021

Takes after previous code by jpeeples67 in https://github.com/GatorSense/Histological_Segmentation

@author: changspencer
"""
## Python standard libraries
from __future__ import print_function
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import utils

## Local external libraries
from .create_dataloaders import Get_Dataloaders
from .models import initialize_model, translate_load_dir
from .metrics import Average_Metric


METRICS_DICT = {
    'Precision': 'Precision',
    'Recall': 'Recall',
    'F1-Score': 'F1',
    'Mean Average Precision': 'mAP',
    'Hausdorff': 'Hausdorff',
    'Jaccard': 'Jaccard',
    'Adjusted Rand': 'Rand',
    'IOU All': 'IOU_All',
    'Pixel Acc': 'Acc',
    'Binary Cross-Entropy': 'BCE',
    'Dice Coefficient': 'Dice_Loss',
    'Specificity': 'Spec'
}

def compute_metrics(input, target, metrics_list=None, pos_wgt=None):
    """
    Computes metrics as determined by the `metrics_list` and what the
        called `Average_Metric` function can handle.
    Args:
        input: array representing the predicted labels for pixels in the original
            images. May be any kind of array that works with `target` but is meant to
            be a learning model's predictions.
        target: array representing the true labels for pixels in `input`
        metrics_list: list of str detailing names of metrics that correspond to
            metrics computed withing `Average_Metric` below.
        pos_wgt: Unused, but originally meant for biasing weight toward positive classes
            due to the imbalanced set of root-soil labels in HyperPRI.
    Returns:
        list containing float values for all desired metrics averaged over the size of input
    """
    all_metrics = list(METRICS_DICT.keys())
    metric_results = {}

    for metric in metrics_list:
        if metric in all_metrics:
            metric_results[metric] = Average_Metric(input.float(),
                                                    target.float(),
                                                    metric_name=METRICS_DICT[metric])
    return metric_results


def eval_models(seg_models, dataloaders, mask_type, device, split, params,
                eval_name="Test", save_dir="Default_Eval/", thresholds:list=None,
                metrics_logged=None, save_mask_png=True, roc_pr=False):
    ''' Evaluate a list of models based on what's given in parameters.
        Saves segmentation masks and computes metrics.
        Currently works for binary segmentation
    Args:
        seg_models: dict of PyTorch model references where keys are their names
        dataloaders: dict of dataloaders where keys are model names
        mask_type: dtype of the ground truth values
        device: PyTorch object for device
        split: int of which run or training split to put the saved files
        params: dict containing a copy of the experimental parameters.
        eval_name: str indicating what to name the figure-saving directory
        save_dir: str that is the full path to contain the figure-saving directory determined
            by `eval_name`
        thresholds: list of floats that correspond to the segmentation threshold
            for each evaluated model. If None, set all to 0.5.
        metrics_logged: list of str the say what metrics should be computed
        save_mask_png: bool flag that indicates whether we should save the masked RGB/HS image
        roc_pr: bool flag that indicates whether to create the precision-recall curve and
            compute its average precision.
    Returns:
        dict of dict's with metrics' average results across a single split/run
    '''
    # Default metrics are all possible metrics (full names)
    if metrics_logged is None:
        metrics_logged = [
            'Precision',
            'Recall',
            'F1-Score',
            'Jaccard',
            'Adjusted Rand',
            'IOU All',
            'Pixel Acc',
            'Binary Cross-Entropy',
            'Dice Coefficient',
            'Specificity'
        ]

    # Default segmentation thresholds are 0.5
    if thresholds is None:
        thresholds = [0.5] * len(seg_models.keys())

    true_vals = {}
    my_preds = {}
    my_metrics = {}
    img_count = 0
    if save_mask_png:
        save_file = save_dir + '{}_Segmentation_Maps/Run_{}/image_iou.txt'.format(
            eval_name,
            split + 1
        )
        # Wipe the old file make a new one...
        with open(save_file, 'w') as f:
            f.write(f"Evaluating models with thresholds: {thresholds}\n\n")

    # Initialize the segmentation model for this run
    for key_idx, model_name in enumerate(seg_models.keys()):

        temp_dataloaders = dataloaders[model_name]

        print(" -- Evaluating {} model...".format(model_name))
        model = seg_models[model_name]
        model = model.to(device)
        
        #Get output and plot
        model.eval()
        for batch in temp_dataloaders:

            imgs, true_masks, idx = (batch['image'], batch['mask'],
                                            batch['index'])

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            for img_idx in range(0, imgs.size(0)):
                # Evaluate each image in the batch
                target = true_masks[img_idx].unsqueeze(0)
                
                img = imgs[img_idx]
                with torch.no_grad():
                    in_img = img.unsqueeze(0)
                    pred = model(in_img)
                    pred_segmap = torch.sigmoid(pred)
                torch.cuda.empty_cache()

                # Save the raw predictions for further analysis (ie. ROC Curve)
                if save_mask_png:
                    # SAVE the prediction mask as another image
                    model_param_str = translate_load_dir(model_name, params)
                    folder = save_dir + '{}_Segmentation_Maps/Run_{}/{}/'.format(
                        eval_name,
                        split + 1,
                        model_param_str
                    )
                    img_name = folder + idx[img_idx]  #+ '_pred.png'

                    # Create Segmentation folders
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    if params['dataset'].lower() == 'hyperpri':  # RGB Data
                        color_correction = img.cpu()
                    else:
                        hsi_rgb = [150, 72, 18]  # R - 700nm, G - 546nm, B - 436nm
                        color_correction = img[hsi_rgb, :, :]**(1 / 2.2)  # Gamma correction
                        color_correction = color_correction.cpu()

                    pred_mask = (pred_segmap > thresholds[key_idx]).float()
                    saved_map = (pred_mask.cpu() * color_correction).squeeze()
                    utils.save_image(saved_map, img_name + "_pred.png")

                if len(my_preds[model_name]) == 0:
                    true_vals[model_name] = target.cpu().flatten().float().numpy()  # Det. random state could decr MEMREQ here
                    my_preds[model_name] = pred_segmap.cpu().flatten().float().numpy()
                else:
                    pred_set = my_preds[model_name]
                    mask_set = true_vals[model_name]
                    my_preds[model_name] = np.concatenate((pred_set, pred_segmap.cpu().flatten().float().numpy()))
                    true_vals[model_name] = np.concatenate((mask_set, target.cpu().flatten().float().numpy()))  # Det. random state could decr MEMREQ here

                # EVAL the prediction mask with various metrics
                pred_segmap = (pred_segmap > thresholds[key_idx]).float()
                eval_results = compute_metrics(pred_segmap, target, metrics_list=metrics_logged)
                if save_mask_png:   #? Deprecated?
                    # Wipe the old file make a new one...
                    with open(save_file, 'a') as f:
                        f.write(f"{model_name} - {idx[img_idx]} - ")
                        f.write(f"+IOU = {eval_results['Jaccard']:.4f}\n")

                if len(my_metrics[model_name]) < 1:
                    my_metrics[model_name] = eval_results
                    fp = ((pred_segmap == 1) * (target == 0)).sum().cpu()
                    my_metrics[model_name]["False Positive"] = fp
                else:
                    # Running average across entire evaluated split/run
                    for metric in metrics_logged:
                        if metric != 'Hausdorff':
                            old_accum = my_metrics[model_name][metric]
                            new_accum = (old_accum * img_count) + eval_results[metric]
                            my_metrics[model_name][metric] = new_accum / (img_count + 1)
                        else:
                            old_accum = my_metrics[model_name][metric]
                            # (n_val-haus_count+1e-7)
                    fp = ((pred_segmap == 1) * (target == 0)).sum().cpu()
                    old_accum = my_metrics[model_name]["False Positive"]
                    new_accum = (old_accum * img_count) + fp
                    my_metrics[model_name]["False Positive"] = new_accum / (img_count + 1)

            img_count += 1
            print("Finished image {} of {} for {}'s dataset".format(img_count,len(dataloaders.sampler), model_name))

        # Force-remove model from memory
        del model

    #? Plot a Precision-Recall Curve for this particular run and for each model
    if roc_pr:
        folder = save_dir + '{}_Segmentation_Maps/Run_{}/'.format(
            eval_name,
            split + 1
        )
        # Create Segmentation folders
        if not os.path.exists(folder):
            os.makedirs(folder)

        plt.figure(dpi=100)
        plt.title(f"Run {split + 1} Test: Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        for key_idx, model_name in enumerate(seg_models.keys()):
            (prec, rec, _) = precision_recall_curve(true_vals[model_name], my_preds[model_name])
            model_ap = average_precision_score(true_vals[model_name], my_preds[model_name])

            # Plot holistic curve
            plt.plot(rec, prec, label=f"{model_name} (AP = {model_ap:.3f})")
        plt.legend()
        pr_fname = f"{folder}pr_curve.png"
        if "SpectralUNET" in seg_models or "CubeNET" in seg_models:
            hsi_hi = params['hsi_hi'] if params['hsi_hi'] > 0 else params['hsi_hi'] + 299
            pr_fname = folder + f"pr_curve_hsi{params['hsi_lo']}_to_{hsi_hi}.png"
        plt.savefig(pr_fname)
        plt.close()

    return my_metrics


def model_pr_stats(seg_models, rel_call_path, mask_type, device, params,
                   save_dir="Default_Eval/", save_mask_png=False, thresholds:dict=None,
                   metrics_logged=None):
    ''' Evaluate a list of models based on what's given in parameters.
        Saves segmentation masks and computes metrics.
        Currently works for binary segmentation.

        Note: Much of this is similar to the 'eval_models' method above except the PR
            curve computation is automated by Scikit-Learn.
    Args:
        seg_models: dict of PyTorch model references where keys are their names
        rel_call_path: str of the full path to the file that called this function. Ideally,
            this is the HyperPRI git repo directory.
        mask_type: dtype of the ground truth values
        device: PyTorch object for device
        params: dict containing a copy of the experimental parameters.
        save_dir: str that is the full path to contain the figure-saving directory determined
            by `eval_name`
        save_mask_png: bool flag that indicates whether we should save the masked RGB/HS image
        thresholds: list of floats that correspond to the segmentation threshold
            for each evaluated model. If None, sets all to 0.5.
        metrics_logged: list of str the say what metrics should be computed
    Returns:
        metrics: dict of dict's with metrics' average results across a single split/run
    '''
    # Default metrics are all possible metrics (full names)
    if metrics_logged is None:
        metrics_logged = [
            'Precision',
            'Recall',
            'F1-Score',
            'Jaccard',
            'IOU All',
            'Pixel Acc',
            'Binary Cross-Entropy',
            'Dice Coefficient',
            'Specificity'
        ]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    #? Plot a Precision-Recall Curve for this particular run and for each model
    folder = save_dir + f'Val_Segmentation_Maps/'
    # Create Segmentation folders
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(dpi=100)
    plt.title(f"Val: Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    for run in range(0, params['splits']):
        # Initialize the prediction/true-val/metric dictionaries at the start
        my_preds = {}
        true_vals = {}
        my_metrics = {}
        for model_key in seg_models.keys():
            my_preds[model_key] = []
            true_vals[model_key] = []
            my_metrics[model_key] = {}

        # Change which dataset we're using for k-fold
        params['json_dir'] = {
            'train': f"{params['imgs_dir']}/data_splits/train{run+1}.json",
            'val': f"{params['imgs_dir']}/data_splits/val{run+1}.json",
            'test': f"{params['imgs_dir']}/data_splits/val{run+1}.json"
        }
        
        print("\n*******************************")
        print("***** VALIDATION - RUN {} *****".format(run + 1))
        print("Using file:", params['json_dir']['val'])

        ## Create dataloaders
        dataloaders = {}
        for model in seg_models:
            params['model_name'] = model
            if model in ["SpectralUNET", "CubeNET"]:
                params['dataset'] = 'HSI_HyperPRI'
            else:
                params['dataset'] = 'HyperPRI'
            print(f"Loading {params['dataset']}...")
            all_dataloaders, pos_class_wt = Get_Dataloaders(0, params, params['batch_size'])
            dataloaders[model] = all_dataloaders['val']
        
        params['pos_class_wt'] = pos_class_wt
        params['n_train'] = len(all_dataloaders['train'].dataset.files)
        params['n_val'] = len(all_dataloaders['val'].dataset.files)
        params['n_test'] = len(all_dataloaders['test'].dataset.files)

        ## Create desired models (using initialize_model) and use pretrained if desired
        for model_key in seg_models.keys():
            if model_key in ["SpectralUNET", "CubeNET"]:
                params['dataset'] = 'HSI_HyperPRI'
                params['channels'] = 1
            else:
                params['dataset'] = "HyperPRI"
                params['channels'] = 3
            params['model_name'] = model_key

            model_param_str = translate_load_dir(model_key, params)
            seg_models[model_key] = initialize_model(model_key, params['num_classes'], params)
            
            params['pretrain_dir'] = f"{rel_call_path}/Saved_Models/{params['dataset']}_finetune/"
            if params['use_pretrained']:
                load_file = params['pretrain_dir'] + model_param_str + f"/Run_{run+1}/best_wts.pt"

            state_dict = torch.load(load_file, map_location="cpu")

            if list(state_dict.keys())[0].startswith('module'):
                new_dict = {}
                query_str = "double_conv"
                for k in state_dict.keys():
                    my_key = k.replace("module.", "", 1)
                    new_dict[my_key] = state_dict[k]
                seg_models[model_key].load_state_dict(new_dict)
            else:
                seg_models[model_key].load_state_dict(state_dict)

        # Evaluate models and add precision-recall curves to the plot...
        for key_idx, model_name in enumerate(seg_models.keys()):
            temp_dataloaders = dataloaders[model_name]

            print("-- Evaluating {} model...".format(model_name))
            model = seg_models[model_name]
            model = model.to(device)
            
            #Get output and plot
            model.eval()
            img_count = 0
            for batch in temp_dataloaders:

                imgs, true_masks, idx = (batch['image'], batch['mask'],
                                                batch['index'])

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=mask_type)
                for img_idx in range(0, imgs.size(0)):
                    # Evaluate each image in the batch
                    target = true_masks[img_idx].unsqueeze(0)
                    
                    img = imgs[img_idx]
                    with torch.no_grad():
                        in_img = img.unsqueeze(0)
                        pred = model(in_img)
                        pred_segmap = torch.sigmoid(pred)
                    torch.cuda.empty_cache()

                    # Save the raw predictions for further analysis (ie. ROC Curve)
                    if save_mask_png and thresholds is not None:
                        # Compute prediction based on best thresholds
                        pred_mask = (pred_segmap > thresholds[model_name][run]).float()

                        if model_name in ["SpectralUNET", "CubeNET"]:
                            params['dataset'] = 'HSI_HyperPRI'
                            params['channels'] = 1
                        else:
                            params['dataset'] = "HyperPRI"
                            params['channels'] = 3
                        params['model_name'] = model_name
                        # SAVE the prediction mask as another image
                        model_param_str = translate_load_dir(model_name, params)
                        pred_folder = f"{folder}/Run_{run+1}/{model_param_str}/"

                        # Create Segmentation folders
                        if not os.path.exists(pred_folder):
                            os.makedirs(pred_folder)

                        img_name = pred_folder + idx[img_idx]
                        utils.save_image(pred_mask, img_name + "_seg.png")
                        
                    elif save_mask_png and thresholds is None:
                        print("Could not save segmentation predictions. Thresholds were not provided...")
                        save_mask_png = False

                    # Generate a PR curve using less memory
                    if len(my_preds[model_name]) == 0:
                        true_vals[model_name] = target.cpu().flatten().float().numpy()  # Det. random state could decr MEMREQ here
                        my_preds[model_name] = pred_segmap.cpu().flatten().float().numpy()
                    else:
                        pred_set = my_preds[model_name]
                        mask_set = true_vals[model_name]
                        my_preds[model_name] = np.concatenate((pred_set, pred_segmap.cpu().flatten().float().numpy()))
                        true_vals[model_name] = np.concatenate((mask_set, target.cpu().flatten().float().numpy()))  # Det. random state could decr MEMREQ here

                img_count += 1
                print(f"   Finished image {img_count} of {len(temp_dataloaders.sampler)} for {model_name}'s dataset", flush=True)

            # Force-remove model from memory
            del model

        dice_threshold = {}
        model_cmat = {}
        for key_idx, model_name in enumerate(seg_models.keys()):
            print(f"{model_name} Precision-Recall...", flush=True)
            (prec, rec, thresh) = precision_recall_curve(true_vals[model_name], my_preds[model_name])
            model_ap = average_precision_score(true_vals[model_name], my_preds[model_name])

            # Determine best threshold based on the PR curve statistics; truncate to 2 decimals
            model_dice = 2 * prec * rec / (prec + rec)
            dice_argmax = model_dice.argmax().astype(int)
            dice_threshold[model_name] = np.trunc(thresh[dice_argmax] * 100) / 100

            # EVAL the prediction mask with various metrics
            all_seg = (my_preds[model_name] > dice_threshold[model_name]).astype(float)

            print("     ...confusion matrices,...", flush=True)
            model_cmat[model_name] = confusion_matrix(true_vals[model_name], all_seg, normalize='true')

            print(f"     ...and metrics", flush=True)

            intersection = (true_vals[model_name] * all_seg).sum()
            union = true_vals[model_name].sum() + all_seg.sum() - intersection + 1e-10

            my_metrics[model_name]['Best Precision'] = prec[dice_argmax]
            my_metrics[model_name]['Best Recall'] = rec[dice_argmax]
            my_metrics[model_name]["Best DICE"] = model_dice.max()
            my_metrics[model_name]["Avg Precision"] = model_ap
            my_metrics[model_name]['+IOU'] = intersection / union

            # Plot holistic curve
            if run == 0:
                label_str = f"{model_name}"
            else:
                label_str = None
            plt.plot(rec, prec, alpha=0.7, c=colors[key_idx], label=label_str)
        
        for model_key in my_metrics.keys():
            key_cmat = model_cmat[model_key]
            print(f"\nBest Threshold for {model_key}: {dice_threshold[model_key]}")
            for metric in my_metrics[model_key]:
                print(f"   {metric} - {my_metrics[model_key][metric]:.4f}")
            print(f"   Confusion Matrix - {key_cmat[0]}")
            print(f"                      {key_cmat[1]}")

    plt.legend()
    pr_fname = f"{folder}pr_cmp_curve.png"
    if "SpectralUNET" in seg_models or "CubeNET" in seg_models:
        hsi_hi = params['hsi_hi'] if params['hsi_hi'] > 0 else params['hsi_hi'] + 299
        pr_fname = folder + f"pr_cmp_curve_hsi{params['hsi_lo']}_to_{hsi_hi}.png"
    plt.savefig(pr_fname)
    plt.close()