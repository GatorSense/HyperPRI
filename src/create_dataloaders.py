# -*- coding: utf-8 -*-
"""
Generate Dataloaders

Adapted from https://github.com/GatorSense/Histological_Segmentation

@author: chang.spencer
"""
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
import random

from .dataset import HyperpriDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def Get_Dataloaders(split, Network_parameters, batch_size):
    
    if Network_parameters['dataset'] == 'HyperPRI':
        train_loader, val_loader, test_loader = load_hyperpri(Network_parameters['imgs_dir'],
                                                          batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          patch_size=Network_parameters['patch_size'],
                                                          rescale=Network_parameters['rescale'],
                                                          json_dir=Network_parameters['json_dir'])

        pos_wt = 1
        
    elif Network_parameters['dataset'] == 'HSI_HyperPRI':
        preserve_cube = Network_parameters['model_name'] == "CubeNET"
        train_loader, val_loader, test_loader = load_hsi_hyperpri(Network_parameters['imgs_dir'],
                                                          batch_size,
                                                          Network_parameters['num_workers'],
                                                          split=split,
                                                          augment=Network_parameters['augment'],
                                                          patch_size=Network_parameters['patch_size'],
                                                          cube_conv=preserve_cube,
                                                          spectral_crop=Network_parameters['model_name'] == 'SpectralUNET',
                                                          hsi_lo=Network_parameters['hsi_lo'],
                                                          hsi_hi=Network_parameters['hsi_hi'],
                                                          json_dir=Network_parameters['json_dir'])

        pos_wt = 1
       
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    return dataloaders, pos_wt


def load_hyperpri(data_path, batch_size, num_workers, pin_memory=True,
                  augment=False, data_subset=None, json_dir:dict={}):
    """
    Args:
        rescale: int stating a resizing factor to match PRMI training.
            Could potentially improve the models' metrics
    """
    # Train data transforms: Resizing and maybe some data augmentation
    if augment:
        train_transform = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
            ]
        )
        # Mask transforms
        gt_transforms = transforms.Compose([transforms.ToTensor()])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(),])
        gt_transforms = transforms.Compose([transforms.ToTensor()])

    # Test transformations
    test_transform = transforms.Compose([transforms.ToTensor(),])
    gt_test_transform = transforms.Compose([transforms.ToTensor()])

    # Have a uniform sampling of classes for each batch
    train_dataset = HyperpriDataset(
        root=data_path, # + "/train",
        img_transform=train_transform,
        label_transform=gt_transforms,
        subset=data_subset,
        json_file=json_dir.get("train", None)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size['train'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        HyperpriDataset(root=data_path, # + "/val",
                     img_transform=test_transform,
                     label_transform=gt_test_transform,
                     subset=data_subset,
                     json_file=json_dir.get("val", None)),
        batch_size=batch_size['val'],
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        HyperpriDataset(root=data_path, # + "/test",
                     img_transform=test_transform,
                     label_transform=gt_test_transform,
                     subset=data_subset,
                     json_file=json_dir.get("test", None)),
        batch_size=batch_size['test'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )
    
    print("Dataloader results: {}, {}, {}".format(len(train_loader),
                                                  len(valid_loader),
                                                  len(test_loader)))
    return train_loader, valid_loader, test_loader


def load_hsi_hyperpri(data_path, batch_size, num_workers, pin_memory=True,
                      patch_size:tuple=(100, 96), data_subset=None, cube_conv=False,
                      spectral_crop=False, hsi_lo=25, hsi_hi=-50, json_dir:dict={}):
    """
    Args:
        rescale: int stating a resizing factor to match PRMI training.
            Could potentially improve the models' metrics
    """
    random_crop = [transforms.RandomCrop(patch_size)]

    # Train data transforms: Deterministic tile crop of the image; additional
    #    augmentations may be added here for "pseudo-data"
    if spectral_crop:   # SpectralUNET
        train_transform = transforms.Compose(random_crop)
        gt_transforms = transforms.Compose(random_crop + [transforms.ToTensor()])
    else:
        train_transform = None
        gt_transforms = transforms.ToTensor()

    # Test transfomrations
    if spectral_crop:   # SpectralUNET
        test_transform = transforms.Compose(random_crop)
        gt_test_transform = transforms.Compose(random_crop + [transforms.ToTensor()])
    else:
        test_transform = None
        gt_test_transform = transforms.ToTensor()

    # Have a uniform sampling of classes for each batch
    train_dataset = HyperpriDataset(
        root=data_path,
        img_transform=train_transform,
        label_transform=gt_transforms,
        subset=data_subset,
        mode='HSI',
        unsqueeze_img=cube_conv,
        hsi_lo=hsi_lo,
        hsi_hi=hsi_hi,
        json_file=json_dir.get('train', None)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size['train'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        HyperpriDataset(root=data_path,
                     img_transform=test_transform,
                     label_transform=gt_test_transform,
                     subset=data_subset,
                     mode='HSI',
                     unsqueeze_img=cube_conv,
                     hsi_lo=hsi_lo,
                     hsi_hi=hsi_hi,
                     json_file=json_dir.get('val', None)),
        batch_size=batch_size['val'],
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        HyperpriDataset(root=data_path,
                     img_transform=test_transform,
                     label_transform=gt_test_transform,
                     subset=data_subset,
                     mode='HSI',
                     unsqueeze_img=cube_conv,
                     hsi_lo=hsi_lo,
                     hsi_hi=hsi_hi,
                     json_file=json_dir.get('test', None)),
        batch_size=batch_size['test'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )
    
    print("Dataloader results: {}, {}, {}".format(len(train_loader),
                                                  len(valid_loader),
                                                  len(test_loader)))
    return train_loader, valid_loader, test_loader
