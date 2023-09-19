# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:46:58 2021

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
import os
import json
import time
import datetime
from collections import OrderedDict
from copy import deepcopy
import pickle
import psutil

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from PIL import Image
import scipy.ndimage as scind
from sklearn.metrics import jaccard_score as jsc
import logging
from barbar import Bar

## PyTorch dependencies
import torch
import torch.nn as nn
from torch import optim
from torchvision import utils
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter

from .metrics import Average_Metric
from .pytorchtools import EarlyStopping

## Local external libraries
from .models import JOSHUA


def save_params(save_dir, params, loader_transforms:dict, model:nn.Module):
    '''
    Print the network parameters to stdout and write to a file
    in the Saved_Results subfolders.
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_params = ['num_classes', 'Splits']
    with open(save_dir + "network_params.txt", "w") as out_file:
        out_file.write('Network Parameters are as follows:\n')
        for key in params.keys():
            key_attr = getattr(params[key], 'keys', None)
            if callable(key_attr):
                key_dict = params[key]
                if key in dataset_params:
                    dataset = params['Dataset']
                    out_file.write(f"   {key}: {key_dict[dataset]}\n")
                else:
                    out_file.write(f"   {key}:\n")
                    for sub_key in key_dict.keys():
                        out_file.write(f"      {sub_key}: {key_dict[sub_key]}\n")
            else:
                out_file.write(f"   {key}: {params[key]}\n")
            
        out_file.write("\nDataloader Transforms\n")
        out_file.write(f"--- Training:\n{loader_transforms['train']}\n\n")
        out_file.write(f"--- Validation:\n{loader_transforms['val']}\n\n")
        out_file.write(f"--- Test:\n{loader_transforms['test']}\n\n")
        out_file.write(f"--- ")
        out_file.write(f"{list(model.children())}")
    
    print("Saved parameters to DIR: " + save_dir)


def translate_net_run_dir(base_dir, run, params):
    if params['model_name'] in ['JOSHUA', 'JOSHUA+', 'JOSHUAres', 'BNH']:
        hist_model = f"{params['hist_model']}"
        filename = f"{base_dir}/{hist_model}/Run_{run + 1}/"
        summaryname = f"{base_dir}/{hist_model}/SummaryRun_{run + 1}/"
    elif params['model_name'] in ['3D_UNET', '3D_BNH']:
        assert params['hsi_lo'] >= 0
        hsi_hi = 299 + params['hsi_hi'] if params['hsi_hi'] <= 0 else params['hsi_hi']
        model_str = f"{params['model_name']}_hsi{params['hsi_lo']}_to_{hsi_hi}"
        filename = f"{base_dir}/{model_str}/Run_{run + 1}/"
        summaryname = f"{base_dir}/{model_str}/SummaryRun_{run + 1}/"
    elif params['model_name'] == 'StatSeg':
        hist_model = f"StatSeg_b{params['numBins']}"
        filename = f"{base_dir}/{hist_model}/Run_{run + 1}/"
        summaryname = f"{base_dir}/{hist_model}/SummaryRun_{run + 1}/"
    elif params['model_name'] == 'SpectralUNET':
        filename = f"{base_dir}/{params['model_name']}_{params['spectral_bn_size']}/Run_{run + 1}/"
        summaryname = f"{base_dir}/{params['model_name']}_{params['spectral_bn_size']}/SummaryRun_{run + 1}/"
    elif params['model_name'] == 'CubeNET':
        filename = f"{base_dir}/{params['model_name']}_{params['3d_featmaps']}/Run_{run + 1}/"
        summaryname = f"{base_dir}/{params['model_name']}_{params['3d_featmaps']}/SummaryRun_{run + 1}/"
    else:   # Baseline model
        filename = f"{base_dir}/{params['model_name']}/Run_{run + 1}/"
        summaryname = f"{base_dir}/{params['model_name']}/SummaryRun_{run + 1}/"

    #Make save directory for previous runs
    now_time = datetime.datetime.now()
    time_str = "_{}{}{}_{}{}{}".format(now_time.year,
                                        now_time.month,
                                        now_time.day,
                                        now_time.hour,
                                        now_time.minute,
                                        now_time.second)
    if not os.path.exists(filename):
        os.makedirs(filename)
    else:
        os.rename(filename, filename[:-1] + time_str + '/')
        os.makedirs(filename)

    if not os.path.exists(summaryname):
        os.makedirs(summaryname)
    else:
        os.rename(summaryname, summaryname[:-1] + time_str + '/')
        os.makedirs(summaryname)

    return filename, summaryname


def hist_bin_dist(net:JOSHUA, device):
    # TODO: Change this to factor in any format of histogram layer implementations
    # TODO (ln 2): For example, all histogram weights should be regularized against their bins' params.
    hist_acc = 0
    if net.pool_hist:
        pool_loc = net.pool_locations
        for idx, loc in enumerate(pool_loc):
            if not loc:  # No histogram pooling
                continue
            down_hist = getattr(net, f"down{idx+1}")
            hist_layer = list(down_hist.pool_conv.children())[1]
            mu = hist_layer.centers.reshape((len(hist_layer.centers) // net.n_bins, net.n_bins))
            sig = hist_layer.widths.reshape((len(hist_layer.widths) // net.n_bins, net.n_bins))

    elif net.parallel_hist:
        skip_loc = net.skip_locations
        for idx, loc in enumerate(skip_loc):
            if not loc:  # No histogram pooling
                continue
            up_hist = getattr(net, f"up{idx+1}")
            hist_layer = list(up_hist.hist_skip.children())[1]
            mu = hist_layer.centers.reshape((len(hist_layer.centers) // net.n_bins, net.n_bins))
            sig = hist_layer.widths.reshape((len(hist_layer.widths) // net.n_bins, net.n_bins))
    else:
        print("No histogram layers...? Skipping regularization...")
        return 0

    mu = mu.unsqueeze(1)
    sig = sig.unsqueeze(2)
    inv_eye = -(torch.eye(net.n_bins) - 1).unsqueeze(0).to(device)
    quad_term = (mu - mu.permute(0, 2, 1))**2

    # Distance is an asymmetric "two-way" RBF
    #   This one constrains both means and deviations
    norm_factor = net.n_bins * (net.n_bins - 1)
    dist_mat = torch.exp(-quad_term * sig**2)
    final_mat = dist_mat * inv_eye / norm_factor

    # # Distance is a symmetric RBF
    # #   This one constrains only means, defaulting to a width of 1
    # norm_factor = net.n_bins * (net.n_bins - 1) / 2
    # dist_mat = torch.exp(-quad_term)
    # final_mat = torch.triu(dist_mat, diagonal=1) / norm_factor

    # # Distance is a symmetric Cauchy/inv.quadratic
    # #   This one only constrains the means
    # norm_factor = net.n_bins * (net.n_bins - 1) / 2
    # dist_mat = 1 / (quad_term + 1)  # scale defaults to 1
    # final_mat = torch.triu(dist_mat, diagonal=1) / norm_factor

    return final_mat.sum() / mu.shape[0]


def hist_div_loss(X1:torch.tensor, X2:torch.tensor, div='chi'):
    # X1 and X2 are B x M, for M pixels
    if div.lower() == 'chi':
        X1 = X1.unsqueeze(1)
        X2 = X2.unsqueeze(1)
        num_quad = (torch.transpose(X1, 1, 2) - X2)**2
        denom_lin = torch.transpose(X1, 1, 2) + X2

        out = torch.sum(num_quad / denom_lin, dim=2).mean()

    elif div.lower() == 'hellinger':
        # Values coming in should be nonzero.
        X1 = torch.sqrt(X1.unsqueeze(1))
        X2 = torch.sqrt(X2.unsqueeze(1))
        bin_diff = (torch.transpose(X1, 1, 2) - X2)**2

        out = torch.sqrt(bin_diff.sum(dim=2)).mean() / torch.sqrt(2.)

    else:
        logging.error("Improperly chosen divergence loss...")
        out = -1

    return out


def train_cycle(net, net_name, train_loader, criterion, optimizer,
                global_step, n_channels, n_classes, device, hist_reg,
                metrics_dict:dict, writer:SummaryWriter, comet_exp=None):
    """
    
    """
    epoch_loss = 0
    epoch_acc = 0
    best_loss = np.inf

    epoch_IOU_pos = 0
    epoch_IOU = 0
    epoch_dice = 0
    epoch_haus_dist = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_f1_score = 0
    epoch_adj_rand = 0
    epoch_map = 0
    epoch_pdf_dist = 0
    inf_samps = 0 #Invalid samples for hausdorff distance

    # t = torch.cuda.get_device_properties(device).total_memory
    # r = torch.cuda.memory_reserved(device)
    # a = torch.cuda.memory_allocated(device)
    # f = r-a  # free inside reserved
    # print("GPU Device memory:", t)
    # print("Starting GPU Memory available:", f)
    
    print(f"(train) Total RAM available: {psutil.virtual_memory()[0] / 10e9}")
    for idx, batch in enumerate(Bar(train_loader)):
        # pdb.set_trace()
        imgs = batch['image']
        true_masks = batch['mask']

        assert imgs.shape[1] == n_channels, \
            f'Network has been defined with {n_channels} input channels, ' \
            f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        imgs = imgs.to(device=device, dtype=torch.float32)  # .contiguous()  # This didn't work?
        mask_type = torch.float32 if n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)

        print(f"(train) RAM Utilization: {psutil.virtual_memory()[2]}")
        # r = torch.cuda.memory_reserved(device)
        # a = torch.cuda.memory_allocated(device)
        # f = r-a  # free inside reserved
        # print("(train) GPU (free) memory:", f)

        masks_pred = net(imgs)
        if net_name in ['StatSeg', 'SegUNET']:
            stats = masks_pred[1]
            masks_pred = masks_pred[0]

            # TODO - Logic for divergence between some number of pixels from root and soil.
            flat_masks = true_masks.view(true_masks.shape[0], -1)
            flat_stats = stats.view(stats.shape[0], stats.shape[1], -1)

            max_pix = 100   # Use only a constant number of pixels for each class
            div_loss = 0
            for stat_idx, stat_pred in enumerate(flat_stats):
                true_root = flat_masks[stat_idx].nonzero()
                true_soil = (-(flat_masks[stat_idx] - 1)).nonzero()
                root_idx = torch.randperm(true_root.shape[0])
                soil_idx = torch.randperm(true_soil.shape[0])
                root_idx = root_idx[:max_pix]
                soil_idx = soil_idx[:max_pix]

                root_pix = stat_pred[:, root_idx]
                soil_pix = stat_pred[:, soil_idx]

                # Compute divergence between same-class, diff-class pixels
                same_class = hist_div_loss(root_pix, root_pix) + hist_div_loss(soil_pix, soil_pix)
                div_loss += same_class.item() - hist_div_loss(root_pix, soil_pix).item()
        else:
            div_loss = 0

        loss = criterion(masks_pred, true_masks) + 0.1 * div_loss
        if hist_reg > 0 and net_name in ["JOSHUA", "JOSHUA+", "JOSHUAres", "BNH"]:
            hist_wgt_reg = hist_bin_dist(net, device)
            loss += hist_reg * hist_wgt_reg

        #Aggregate loss for epoch
        epoch_loss += loss.item() * imgs.size(0)
    
        pred_out = (torch.sigmoid(masks_pred.detach()) > .5).float()  #! Should this be changed...?
        temp_haus, temp_haus_count = Average_Metric(pred_out, 
                                                    true_masks,
                                                    metric_name='Hausdorff')
        epoch_haus_dist += temp_haus
        epoch_dice += Average_Metric(pred_out, true_masks, metric_name="Dice_Loss")
        epoch_IOU_pos += Average_Metric(pred_out, true_masks,metric_name='Jaccard')
        epoch_IOU += Average_Metric(pred_out, true_masks,metric_name='IOU_All')
        epoch_acc += Average_Metric(pred_out, true_masks,metric_name='Acc')
        inf_samps += temp_haus_count
        epoch_prec += Average_Metric(pred_out, true_masks,metric_name='Precision')
        epoch_rec += Average_Metric(pred_out, true_masks,metric_name='Recall')
        epoch_f1_score += Average_Metric(pred_out, true_masks,metric_name='F1')
        epoch_adj_rand += Average_Metric(pred_out, true_masks,metric_name='Rand')
        # epoch_map += Average_Metric(pred_out, true_masks,metric_name='mAP')
        epoch_pdf_dist += div_loss
        epoch_map += 0

        # Clear out memory, etc.
        del true_masks
        del imgs
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(net.parameters(), 0.1)  #! Should I remove...?
        optimizer.step()

    total = len(train_loader.sampler)
    metrics_dict = {}

    writer.add_scalar('Dice/train', epoch_dice/total, global_step)
    writer.add_scalar('IOU_pos/train',epoch_IOU_pos/total,global_step)
    writer.add_scalar('Loss/train', epoch_loss, global_step)
    writer.add_scalar('Pixel_Acc/train',epoch_acc/total,global_step)
    writer.add_scalar('Overall_IOU/train',epoch_IOU/total,global_step)
    writer.add_scalar('HausdorffDistance/train',epoch_haus_dist/(total-inf_samps+1e-7),global_step)
    writer.add_scalar('adj_rand/train',epoch_adj_rand/total,global_step)
    writer.add_scalar('precison/train',epoch_prec/total,global_step)
    writer.add_scalar('recall/train',epoch_rec/total,global_step)
    writer.add_scalar('f1_score/train',epoch_f1_score/total, global_step)
    writer.add_scalar('mAP/train', epoch_map / total, global_step)
    writer.add_scalar('pdf_dist/train', epoch_pdf_dist / total, global_step)

    metrics_dict['Dice_F1'] = epoch_dice / total
    metrics_dict['IOU_pos'] = epoch_IOU_pos / total
    metrics_dict['Loss'] = epoch_loss
    metrics_dict['Pixel_Acc'] = epoch_acc / total
    metrics_dict['Overall_IOU'] = epoch_IOU / total
    metrics_dict['HausdorffDistance'] = epoch_haus_dist / (total-inf_samps+1e-7)
    metrics_dict['adj_rand'] = epoch_adj_rand / total
    metrics_dict['precison'] = epoch_prec / total
    metrics_dict['recall'] = epoch_rec / total
    metrics_dict['f1_score'] = epoch_f1_score / total
    metrics_dict['mAP'] = epoch_f1_score / total
    metrics_dict['pdf_dist'] = epoch_pdf_dist / total
    
    print('train Loss: {:.4f} IOU_pos: {:.4f} Dice Coefficient: {:.4f}'.format(epoch_loss/total, 
                                                                               epoch_IOU_pos/total,
                                                                               epoch_dice/total))
    if comet_exp is not None:
        comet_exp.log_metrics(metrics_dict, epoch=global_step - 1)

    return metrics_dict


def validation_cycle(net, net_name, val_loader, pos_wt, global_step, device,
                     hist_reg, writer:SummaryWriter, comet_exp=None):
    """
    
    """
    for tag, value in net.named_parameters():
        if value.requires_grad:
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

    val_dict = eval_net(net, net_name,
                        val_loader,
                        device, hist_reg,
                        pos_wt=torch.tensor(pos_wt))
    if comet_exp is not None:
        comet_exp.log_metrics(val_dict, epoch=global_step - 1)
    
    logging.info('Validation Dice Coeff: {}'.format(val_dict['dice']))

    writer.add_scalar('Dice/val', val_dict['dice'], global_step)
    writer.add_scalar('IOU_pos/val',val_dict['pos_IOU'], global_step)
    writer.add_scalar('Loss/val', val_dict['loss'],  global_step)
    writer.add_scalar('Pixel_Acc/val',val_dict['pixel_acc'], global_step)
    writer.add_scalar('Overall_IOU/val',val_dict['overall_IOU'], global_step)
    writer.add_scalar('HausdorffDistance/val',val_dict['haus_dist'], global_step)
    writer.add_scalar('adj_rand/val',val_dict['adj_rand'], global_step)
    writer.add_scalar('precison/val',val_dict['precision'], global_step)
    writer.add_scalar('recall/val',val_dict['recall'], global_step)
    writer.add_scalar('f1_score/val',val_dict['f1_score'], global_step)
    writer.add_scalar('f1_score/val',val_dict['mAP'], global_step)

    return val_dict


def train_model(net, dataloaders, train_params:dict,
                device, split, thresholds=None, save_epoch=5, 
                save_dir="Saved_Models/", dir_checkpoint='checkpoints/',
                save_cp=False, comet_exp=None, ckpt_data:dict=None):
    ''' Train a list of models based on what's given in parameters.
        Saves training/validation metrics.
        Currently works for binary segmentation
    Args:
        net: PyTorch module of learning model to be trained
        dataloaders: dict of dataloaders where keys are 'train', 'val', 'test'
        train_params:
        device: PyTorch object for device
        split: int of which run or training split to put the saved files
        thresholds: list of floats that correspond to the segmentation threshold
            for each evaluated model. If None, set all to 0.5.
        save_epoch: list of str the say what metrics should be computed
        save_dir:
        dir_checkpoint:
        save_cp:
        comet_exp:
    Returns: None
    '''
    # writer = SummaryWriter(log_dir=sum_name+ 'Run_' +str(split+1))
    since = time.time()

    logging.info(f'''Starting training:
        Epochs:                {train_params['epochs']}
        Training Batch size:   {train_params['batch_size']['train']}
        Validation Batch size: {train_params['batch_size']['val']}
        Test Batch size:       {train_params['batch_size']['test']}
        Optimizer:             {train_params['optim']}
        Learning rate:         {train_params['lr']}
        Weight Decay:          {train_params['wgt_decay']}
        Momentum (SGD only):   {train_params['momentum']}
        Early Stopping:        {train_params['early_stop']}
        Training size:         {train_params['n_train']}
        Validation size:       {train_params['n_val']}
        Testing size:          {train_params['n_test']}
        Checkpoints:           {save_cp}
        Device:                {device.type}
    ''')
    net_name = train_params['model_name']
    num_epochs = train_params['epochs']
    pos_wt = train_params['pos_class_wt']

    if not ckpt_data:
        dir_name, summ_name = translate_net_run_dir(save_dir, split, train_params)
    else:
        dir_name = ckpt_data['dir_name']
        summ_name = ckpt_data['summ_name']

    save_params(dir_name, train_params,
        {
            'train': dataloaders['train'].dataset.img_transform,
            'val': dataloaders['val'].dataset.img_transform,
            'test': dataloaders['test'].dataset.img_transform
        },
        net
    )

    writer = SummaryWriter(log_dir=summ_name)

    # Set optimizer
    if train_params['optim'] == 'sgd':
        optimizer = optim.SGD(net.parameters(),
                              lr=train_params['lr'],
                              weight_decay=train_params['wgt_decay'],
                              momentum=train_params['momentum'])
    elif train_params['optim'] == 'adamax':
        optimizer = optim.Adamax(net.parameters(), lr=train_params['lr'],
                                 weight_decay=train_params['wgt_decay'])
    else:
        optimizer = optim.Adam(net.parameters(), lr=train_params['lr'],
                               weight_decay=train_params['wgt_decay'])
    
    #Set Early stopping
    early_stopping = EarlyStopping(patience=train_params['early_stop'], verbose=True)
    
    # The following currently does not work for the multi_GPU case
    # if torch.cuda.device_count() > 1:
    #     n_classes = net.module.n_classes
    #     n_channels = net.module.n_channels
    # else:
    n_classes = net.n_classes
    n_channels = net.n_channels
    
    # Criterion set for 1 segmentation class
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_wt).to(device))

    if ckpt_data:
        if os.path.exists(dir_name + dir_checkpoint + f'optimizer.pth'):
            optim_state = torch.load(dir_name + dir_checkpoint + f'optimizer.pth')
            optimizer.load_state_dict(optim_state)
            print("   (Loaded optimizer state)")

        start_ep = ckpt_data['epoch'] + 1
        global_step = ckpt_data['global_step']
        best_dice = ckpt_data['best_dice']

        loaded_dice = np.array(ckpt_data['val_dice_track'])
        max_pad = max(0, num_epochs - len(loaded_dice))
        val_dice_track = np.pad(loaded_dice, ((0, max_pad)))
        
        # Load early stopping information
        for attr in ckpt_data['early_stop']:
            setattr(early_stopping, attr, ckpt_data['early_stop'][attr])
    else:
        start_ep = 0
        global_step = 1
        best_dice = -np.inf
        val_dice_track = np.zeros(num_epochs)
    
    train_dict = {
        'Dice_F1': 0.0,
        'IOU_pos': 0.0,
        'Loss': 0.0,
        'Pixel_Acc': 0.0,
        'Overall_IOU': 0.0,
        'HausdorffDistance': 0.0,
        'adj_rand': 0.0,
        'precison': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'mAP': 0.0,
        'pdf_dist': 0.0,
    }
    
    val_dict = {
        'Dice_F1': 0.0,
        'IOU_pos': 0.0,
        'Loss': 0.0,
        'Pixel_Acc': 0.0,
        'Overall_IOU': 0.0,
        'HausdorffDistance': 0.0,
        'adj_rand': 0.0,
        'precison': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'mAP': 0.0,
        'pdf_dist': 0.0,
    }
    
    for epoch in range(start_ep, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        net.train()
        val_iter_track = []

        for phase in ['train','val']:
            
            if phase == 'train':
                net.train()
                with None if comet_exp is None else comet_exp.train() as exp:
                    train_dict = train_cycle(net, net_name, dataloaders[phase], criterion, optimizer,
                                             global_step, n_channels, n_classes,
                                             device, train_params['hist_reg'], train_dict, writer, comet_exp)
            else:
                net.eval()
            
                with None if comet_exp is None else comet_exp.validate() as exp:
                    val_dict = validation_cycle(net, net_name, dataloaders[phase], pos_wt, global_step,
                                                device, train_params['hist_reg'], writer, comet_exp)
    
                val_iter_track.append(val_dict['dice'])
                global_step += 1
                print('val Loss: {:.4f} IOU_pos: {:.4f} Dice Coefficient: {:.4f} Avg Inf Time: {:4f}s'.format(
                    val_dict['loss'], 
                    val_dict['pos_IOU'],
                    val_dict['dice'],
                    val_dict['inf_time']
                ))
                # deprecated - Average over last 100 epochs (3D-UNET instability motivated me to do this)
                #    ie. the model must do better across 100 epochs to overwrite the best weights
                val_dice_track[epoch] = sum(val_iter_track[-100:]) / len(val_iter_track[-100:])
                early_stopping(val_dict['loss'], net)

        # Check dice coefficient and save best model
        if val_iter_track[-1] > best_dice:
            best_dice = val_iter_track[-1]
            best_wts = deepcopy(net.state_dict())
            val_metrics = val_dict

            print(f"Epoch {epoch}: Saving best {train_params['model_name']} model...")
            torch.save(best_wts, dir_name + 'best_wts.pt')
            torch.save(optimizer.state_dict(), dir_name + f'best_optim_state.pth')
            # Save extraneous checkpoint info: best dice, epoch, early stopping, etc.
            early_stop_attrs = { attr:getattr(early_stopping, attr) for attr in dir(early_stopping)
                                    if not callable(getattr(early_stopping, attr)) and "__" not in attr}
            best_ckpt = {
                # "optim": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "val_dice_track": list(val_dice_track),
                "early_stop": early_stop_attrs,
                "best_dice": best_dice,
                "params": train_params,
                "dir_name": dir_name,
                "summ_name": summ_name
            }
            # np.save(f"{dir_name}/best_dice_track.npy", val_dice_track)
            with open(f"{dir_name}/best_ckpt_info.json", 'w') as file:
                json.dump(best_ckpt, file)

        #Save every save_epoch
        if save_cp and (epoch+1) % save_epoch == 0:
            try:
                os.mkdir(dir_name + dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_name + dir_checkpoint + f'checkpoint.pth')
            torch.save(optimizer.state_dict(), dir_name + dir_checkpoint + f'optimizer.pth')
            logging.info(f'Checkpoint saved at epoch {epoch + 1}!')

            # Save extraneous checkpoint info: best dice, epoch, early stopping, etc.
            early_stop_attrs = { attr:getattr(early_stopping, attr) for attr in dir(early_stopping)
                                    if not callable(getattr(early_stopping, attr)) and "__" not in attr}
            ckpt = {
                # "optim": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "val_dice_track": list(val_dice_track),
                "early_stop": early_stop_attrs,
                "best_dice": best_dice,
                "params": train_params,
                "dir_name": dir_name,
                "summ_name": summ_name
            }
            # np.save(f"{dir_name}{dir_checkpoint}/val_dice_track.npy", val_dice_track)
            with open(f"{dir_name}{dir_checkpoint}/ckpt_info.json", 'w') as file:
                json.dump(ckpt, file)

        # Early stop once loss stops improving
        if early_stopping.early_stop:
            print()
            print("Early stopping")
            break

    time_elapsed = time.time() - since
    text_file = open(dir_name + 'Run_Time.txt','w')
    n = text_file.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    text_file.close()
    text_file = open(dir_name + 'Training_Weight.txt','w')
    n = text_file.write('Training Positive Weight: ' + str(pos_wt))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    if best_wts:
        torch.save(best_wts, dir_name + 'best_wts.pt')
    output_val = open(dir_name + 'val_metrics.pkl','wb')
    pickle.dump(val_metrics, output_val)
    output_val.close()
    writer.close()


def eval_net(net, net_name, loader, device, hist_reg,
             pos_wt=torch.tensor(1),
             best_wts=None):
    """
    Generalized Evaluation of a network given data in the loader
    """
    if best_wts is not None:
        net.load_state_dict(best_wts)
    net.eval()
    try:
        mask_type = torch.float32 if net.module.n_classes == 1 else torch.long
    except:
        mask_type = torch.float32 if net.n_classes == 1 else torch.long

    n_val = 0
    dice_tot = 0
    jacc_score = 0
    loss = 0
    inf_time = 0
    iou_score = 0
    class_acc = 0
    haus_dist = 0
    haus_count = 0
    prec = 0
    rec = 0
    f1_score = 0
    adj_rand = 0
    mAP = 0
    spec = 0
    pdf_dist = 0

    # t = torch.cuda.get_device_properties(device).total_memory
    # r = torch.cuda.memory_reserved(device)
    # a = torch.cuda.memory_allocated(device)
    # f = r-a  # free inside reserved
    # print("GPU Device memory:", t)
    # print("(val) Starting GPU Memory available:", f)
    
    print(f"(eval) Total RAM available: {psutil.virtual_memory()[0] / 10e9}")
    for idx, batch in enumerate(Bar(loader)):
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        # r = torch.cuda.memory_reserved(device)
        # a = torch.cuda.memory_allocated(device)
        # f = r-a  # free inside reserved
        # print("GPU (free) memory:", f)

        print(f"(eval) RAM Utilization: {psutil.virtual_memory()[2]}")
        with torch.no_grad():
            temp_start_time = time.time()
            mask_pred = net(imgs)
            temp_end_time = (time.time() - temp_start_time)/imgs.size(0)
            inf_time += temp_end_time

            if net_name in ['StatSeg', 'SegUNET']:
                stats = mask_pred[1]
                mask_pred = mask_pred[0]

                # TODO - Logic for divergence between some number of pixels from root and soil.
                flat_masks = true_masks.view(true_masks.shape[0], -1)
                flat_stats = stats.view(stats.shape[0], stats.shape[1], -1)

                max_pix = 100   # Use only a constant number of pixels for each class
                div_loss = 0
                for stat_idx, stat_pred in enumerate(flat_stats):
                    true_root = flat_masks[stat_idx].nonzero()
                    true_soil = (-(flat_masks[stat_idx] - 1)).nonzero()
                    root_idx = torch.randperm(true_root.shape[0])
                    soil_idx = torch.randperm(true_soil.shape[0])
                    root_idx = root_idx[:max_pix]
                    soil_idx = soil_idx[:max_pix]

                    root_pix = stat_pred[:, root_idx]
                    soil_pix = stat_pred[:, soil_idx]

                    # Compute divergence between same-class, diff-class pixels
                    same_class = hist_div_loss(root_pix, root_pix).sum() + hist_div_loss(soil_pix, soil_pix).sum()
                    div_loss += same_class - hist_div_loss(root_pix, soil_pix).sum().item()
            else:
                div_loss = 0
            pred = torch.sigmoid(mask_pred)

            loss += Average_Metric(pred,true_masks,pos_wt=pos_wt.to(device),metric_name='BCE') + 0.1 * div_loss
            # loss += Average_Metric(pred,true_masks,pos_wt=pos_wt.to(device),metric_name='BCE')

            if hist_reg > 0 and net_name in ["JOSHUA", "JOSHUA+", "JOSHUAres", "BNH"]:
                hist_wgt_reg = hist_bin_dist(net, device)
                loss += hist_reg * hist_wgt_reg

            pred = (pred > 0.5).float()
            prec += Average_Metric(pred, true_masks,metric_name='Precision')
            rec += Average_Metric(pred, true_masks,metric_name='Recall')
            f1_score += Average_Metric(pred, true_masks,metric_name='F1')
            temp_haus, temp_haus_count = Average_Metric(pred, true_masks,metric_name='Hausdorff')
            haus_dist += temp_haus
            haus_count += temp_haus_count
            jacc_score += Average_Metric(pred, true_masks, metric_name='Jaccard')
            dice_tot += Average_Metric(pred, true_masks, metric_name="Dice_Loss")
            adj_rand += Average_Metric(pred, true_masks, metric_name='Rand')
            iou_score += Average_Metric(pred, true_masks, metric_name='IOU_All')
            class_acc += Average_Metric(pred, true_masks, metric_name='Acc')
            spec += Average_Metric(pred, true_masks, metric_name='Spec')
            pdf_dist += div_loss
            # mAP += Average_Metric(pred, true_masks, metric_name="mAP")
            mAP += 0
            n_val += true_masks.size(0)

        del imgs
        del true_masks

    metrics = {
        'dice': dice_tot / n_val,
        'pos_IOU': jacc_score / n_val,
        'loss': (loss / n_val),
        'inf_time': inf_time / n_val,
        'overall_IOU': iou_score/ n_val,
        'pixel_acc': class_acc / n_val,
        'haus_dist': haus_dist / (n_val-haus_count + 1e-7),
        'adj_rand': adj_rand / n_val,
        'precision': prec / n_val,
        'recall': rec / n_val,
        'f1_score': f1_score / n_val, 
        'specificity': spec/ n_val,
        'mAP': mAP / n_val,
        'pdf_dist': pdf_dist / n_val
    }
        
    return metrics


def post_process(net, data_loader, net_params:dict, thresh=0.5, roots=True):
    '''
    Utilize the thinning and eroding operations to improve the precision
        of either roots or soil pixels.
    Root operations: 
    Soil operations: 

    Arguments:
    net - torch.nn.Module used to predict segmentation on dataset
    data_loader - torch.data.utils.DataLoader that allows iteration through
            image data and there respective segmentation masks
    net_params - dict containing all necessary pathing parameters for
            saving the post-processed segmentation images
    thresh - float used for Heaviside thresholding of segmentation 'confidence'
    roots - bool stating which class we want to segment with high precision
            (eg. roots or soil)
    '''
    #! #! #! EROSION IMAGE TRANSFORMATION - Acquire root pixels !# !# !#
    # length = 13
    # diam = 2
    # B1 = np.ones((length, diam))
    # B2 = np.ones((diam, length))
    # B3 = scind.rotate(B1, -45)  #np.ones((2, 7))
    # B4 = scind.rotate(B2, -45)  #np.ones((2, 7))
    # B5 = np.ones((3, 3))

    # precision, recall, dice, _ = precision_recall_fscore_support(og_mask.flatten(), pred.flatten(), labels=[1])
    # print(f"Original Precision: {precision[0]:.3f}")
    # print(f"Original Recall: {recall[0]:.3f}")
    # print(f"Original Dice: {dice[0]:.3f}\n---------------------")

    # #? Open and close, open and close, open...
    # eroded1 = scind.binary_opening(pred, B1)
    # eroded1 = scind.binary_closing(eroded1, B1)
    # eroded2 = scind.binary_opening(pred, B2)
    # eroded2 = scind.binary_closing(eroded2, B2)
    # eroded3 = scind.binary_opening(pred, B3)
    # eroded3 = scind.binary_closing(eroded3, B3)
    # eroded4 = scind.binary_opening(pred, B4)
    # eroded4 = scind.binary_closing(eroded4, B4)

    # #? OR operation on thinning results
    # eroded = eroded1 + eroded2 + eroded3 + eroded4
    # eroded = (eroded > 0) * 1

    # #? Erode the image with the final structural element
    # eroded = scind.binary_erosion(eroded, B5)
    return None