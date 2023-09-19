# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:46:58 2021

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import jaccard_score as jsc
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from .models import initialize_model, translate_load_dir

def Generate_Dir_Name(split,Network_parameters):
    
    if Network_parameters['hist_model'] is not None:
        dir_name = (Network_parameters['folder'] + '/' + Network_parameters['mode'] 
                    + '/' + Network_parameters['dataset'] + '/' 
                    + Network_parameters['hist_model'] + '/Run_' 
                    + str(split + 1) + '/')
    #Baseline model
    else:
        dir_name = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['dataset'] + '/' +
                    Network_parameters['model'] 
                    + '/Run_' + str(split + 1) + '/')  
    
    #Location to save figures
    fig_dir_name = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['dataset'] + '/')
        
    return dir_name, fig_dir_name

def inverse_normalize(tensor, mean=(0,0,0), std=(1,1,1)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def Generate_Images(dataloaders, mask_type, model_names, device, split,
                    num_classes, params, alpha=.35, use_cuda=False):
    def find_best_weights(model_param_str):
        sub_dir = params['pretrain_dir'] + model_param_str + f"/Run_{split+1}"
        state_dict = torch.load(f"{sub_dir}/best_wts.pt", map_location=device)

        return state_dict

    for phase in ['val']:  #,'test']:
        print(f"{phase} Image Phase: {len(dataloaders[phase])} images")
        img_count = 0
        for batch in dataloaders[phase]:
           
            imgs, true_masks, idx = (batch['image'], batch['mask'],
                                              batch['index'])
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
           
            for img in range(0, imgs.size(0)):
            # for img in range(0, 1):  # Limit number of outputs to expedite eval
        
                #Create figure for each image
                temp_fig, temp_ax = plt.subplots(nrows=1,
                                                 ncols=len(model_names) + 2,
                                                 dpi=300)
                # Un-normalize the image to put in correct colors
                cpu_img = imgs[img].cpu()

                #Plot images, hand labels, and masks
                temp_ax[0].imshow(cpu_img.permute(1, 2, 0))
                temp_ax[0].tick_params(axis='both', labelsize=0, length = 0)
                # Only one class (binary segmentation)
                temp_ax[1].imshow(cpu_img.permute(1,2,0))
                M, N = true_masks[img][0].shape
                temp_overlap = np.zeros((M,N,3))
                gt_mask = true_masks[img][0].cpu().numpy().astype(dtype=bool)
                temp_overlap[gt_mask,:] = [5/255, 133/255, 176/255]
                temp_ax[1].imshow(temp_overlap,'jet',interpolation=None,alpha=alpha)
                temp_ax[1].tick_params(axis='both', labelsize=0, length = 0)

                axes = temp_ax

                #Labels Rows and columns
                if num_classes == 1:
                    col_names = [idx[img], 'Ground Truth'] + model_names
                else:
                    col_names = ['Input Image', 'Ground Truth'] + model_names
                cols = ['{}'.format(col) for col in col_names]
                
                for ax, col in zip(axes, cols):
                    ax.set_title(col)
            
                # Initialize the segmentation model for this run
                for key_idx, eval_model in enumerate(model_names):
                    model_param_str = translate_load_dir(eval_model, params)
                    if model_param_str == 'PrmiNET':
                        model_param_str = "XuNET"
                    model = initialize_model(eval_model, params['num_classes'], params)
                    fig_dir = '{}/{}_Segmentation_Maps/Run_{}/'.format(params['pretrain_dir'],
                                                                       phase.capitalize(),
                                                                       split+1)

                    print(" -- Evaluating {} model...".format(eval_model))

                    #If parallelized, need to set model
                      # Send the model to GPU if available
                    # if use_cuda and torch.cuda.device_count() > 1:
                    #     print("Using", torch.cuda.device_count(), "GPUs!")
                    #     model = nn.DataParallel(model)

                    model = model.to(device)
                    best_wts = find_best_weights(model_param_str)
                    model.load_state_dict(best_wts)
                    
                    #Get output and plot
                    model.eval()
                    
                    with torch.no_grad():
                        preds = model(imgs[img].unsqueeze(0))
                    
                    preds = (torch.sigmoid(preds) > params['thresholds'][key_idx]).float()
                    
                    #Plot masks only
                    M, N = true_masks[img][0].shape
                    temp_overlap = np.zeros((M,N,3))
                    preds_mask = preds[0].cpu().permute(1,2,0)[:,:,0].numpy().astype(dtype=bool)
                    gt_mask = true_masks[img][0].cpu().numpy().astype(dtype=bool)
                    temp_overlap[:,:,0] = preds_mask
                    temp_overlap[:,:,1] = gt_mask
                    
                    #Convert to color blind
                    #Output
                    temp_overlap[preds_mask,:] = [202/255, 0/255, 32/255]  # Red
                    temp_overlap[gt_mask, :] = [5/255, 133/255, 176/255]   # Blue
                    agreement = preds_mask * gt_mask
                    temp_overlap[agreement, :] = [155/255, 191/255, 133/255]  # Green
                    
                    temp_ax[key_idx+2].imshow(cpu_img.permute(1,2,0))
                    temp_ax[key_idx+2].imshow(temp_overlap,alpha=alpha)
                    temp_ax[key_idx+2].tick_params(axis='both', labelsize=0, length = 0)

                    # Save the color-coded prediction as a separate image
                    model_fig = plt.figure(dpi=400)
                    plt.title(eval_model)
                    plt.imshow(cpu_img.permute(1,2,0))
                    plt.imshow(temp_overlap, alpha=alpha)
                    plt.tick_params(axis='both', labelsize=0, length = 0)
                    model_fig.savefig(f"{fig_dir}/{model_param_str}/{idx[img]}_seg.png", 
                                      dpi=temp_fig.dpi, bbox_inches='tight')
                    plt.close(model_fig)

                    del model
                    torch.cuda.empty_cache()
                
                #Create Training and Validation folder
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                 
                img_name = fig_dir + idx[img] + '.png'
                
                temp_fig.savefig(img_name, dpi=temp_fig.dpi, bbox_inches='tight')
                plt.close(fig=temp_fig)
            
                img_count += 1
                print('Finished image {} of {} for {} dataset'.format(img_count,len(dataloaders[phase].sampler),phase))
