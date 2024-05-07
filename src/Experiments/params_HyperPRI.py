import os
import datetime
import sys
import numpy as np

import torch
import torchvision.transforms as transforms

# TODO List
# how to set parameters
# how to load parameters after saving
from .models import *
from ..dataset import HyperpriDataset

class ExpRedGreenBluePRI:
    """
    Experimental parameters for training models on RGB HyperPRI data
    """
    def __init__(self, calling_path, split_no=1, seed_num=0, augment=False, comet_logging=True):
        #Define class attributes (actual paramters)
        self.now = datetime.datetime.now()

        # Basic definitions -----------------------------------------------
        self.dataset = "RGB"
        self.b_size  = {'train': 2, 'val': 2, 'test': 1}
        self.device  = 'gpu'
        self.epochs  = 2000

        # Dataset Definitions ---------------------------------------------
        self.patch_size  = (608, 968)
        self.color_mode  = 'rgb'
        self.channels    = 3 if self.color_mode.lower() != 'gray' else 1
        self.rescale     = 1
        self.augment     = augment
        self.rotate      = False
        self.num_classes = 1
        self.label_set   = None   # Set to 'None' for normal root-soil segmentation
        self.data_dir    = f"{calling_path}/Datasets/HyperPRI/"
        self.json_dir    = {
            'train': f"{self.data_dir}/data_splits/train{split_no}.json",
            'val': f"{self.data_dir}/data_splits/val{split_no}.json",
            'test': f"{self.data_dir}/data_splits/val{split_no}.json"
        }
        self.run_num    = 10 * seed_num + split_no  # If deciding to run multiple seeds per data split...

        self.train_transforms   = [transforms.RandomCrop(self.patch_size), transforms.ToTensor()]
        self.gt_transforms      = [transforms.RandomCrop(self.patch_size), transforms.ToTensor()]
        self.test_transforms    = [transforms.ToTensor(),]
        self.gt_test_transforms = [transforms.ToTensor()]

        # Model Parameters ------------------------------------------------
        self.model_name         = "UNET"
        self.bilinear           = False
        self.feature_extraction = False
        self.use_attention      = False
        self.use_attention      = False
        self.use_pretrained     = False

        # PyTorch required parameters -------------------------------------
        self.criterion          = torch.nn.BCEWithLogitsLoss()
        self.optimizer          = "adam"
        self.learn_rate         = 0.001
        self.weight_decay       = 0
        self.momentum           = 0.9
        self.test_deepspeed     = None

        # PyLightning Metrics param's -------------------------------------
        self.task         = "binary"
        self.threshold    = 0.5

        # Early Stopping parameters ---------------------------------------
        self.consecutive = None
        self.overall     = 500

        # Visualization parameters ----------------------------------------
        self.model_param_str = self.translate_load_dir()
        self.save_path       = f"{calling_path}/Saved_Models/{self.dataset}/{self.model_param_str}/Run_{self.run_num}/"
        self.fig_dir         = f"{calling_path}/Saved_Models/{self.dataset}/Val_Segmentation_Maps/Run_{self.run_num}/{self.model_param_str}/"

        # Comet Logging Parameters ----------------------------------------
        self.comet_params = {
            "api_key": os.environ.get("COMET_API_KEY") if comet_logging else None,
            "workspace": os.environ.get("COMET_WORKSPACE") if comet_logging else None,
            "offline_dir": f"{calling_path}/comet_offline/",
            "project_name": "hyperpri",
            "experiment_name": f"{self.dataset}-{self.model_name}-{self.run_num}",
        }

    def change_network_param(self, new_model_name:str, calling_path:str,
                             split_no:int, seed_num=0, model_params:dict=None):
        """
        For evaluating multiple models and
        changing the parameters' model info on-the-fly
        """
        # If a dictionary of parameters is provided, overwrite the provided param's
        if model_params is not None:
            for k_idx, k in enumerate(model_params):
                this_attr = getattr(self, k, None)
                if this_attr is not None:
                    setattr(self, k, model_params[k])

        self.run_num = 10 * seed_num + split_no
        self.model_name = new_model_name
        self.model_param_str = self.translate_load_dir()
        self.save_path = f"{calling_path}/Saved_Models/{self.dataset}/{self.model_param_str}/Run_{self.run_num}/"
        self.fig_dir   = f"{calling_path}/Saved_Models/{self.dataset}/Val_Segmentation_Maps/Run_{self.run_num}/{self.model_param_str}/"

    def translate_load_dir(self):    #Generate segmentation model
        if self.model_name.lower() in ['unet', 'unet+']:
            model_str = self.model_name
        else:
            err_str = f"{self.model_name} is not in list of possible models\n"
            err_str += "   (accepted: UNET, UNET+)"
            raise ValueError(err_str)
        return model_str

    def get_network(self):
        if self.model_name.lower() in ['unet', 'unet+']:
            model = UNet(self.channels,
                         self.num_classes,
                         bilinear=self.bilinear,
                         feature_extraction=self.feature_extraction,
                         use_attention=self.use_attention)
        else: #Show error that segmentation model is not available
            raise RuntimeError('ExpRedGreenBluePRI: Invalid model')

        return model

    def get_train_data(self):
        return HyperpriDataset(root=self.data_dir, # + "/train",
                               mode=self.color_mode,
                               img_transform=transforms.Compose(self.train_transforms),
                               label_transform=transforms.Compose(self.gt_transforms),
                               subset=self.label_set,
                               json_file=self.json_dir.get("train", None)
        )

    def get_val_data(self):
        return HyperpriDataset(root=self.data_dir,
                               img_transform=transforms.Compose(self.test_transforms),
                               label_transform=transforms.Compose(self.gt_test_transforms),
                               subset=self.label_set,
                               json_file=self.json_dir.get("val", None)
        )

    def get_test_data(self):
        return HyperpriDataset(root=self.data_dir,
                               img_transform=transforms.Compose(self.test_transforms),
                               label_transform=transforms.Compose(self.gt_test_transforms),
                               subset=self.label_set,
                               json_file=self.json_dir.get("test", None)
        )

    def get_test_id(self):
        '''Creates name to identify experiment output files/folders by'''

        # Create unique identifier for test
        testID = '%d%02d%02d_%02d%02d' % (self.now.year, self.now.month, self.now.day, self.now.hour, self.now.minute)

        lrstring = str(self.learningRate).replace('.','-')
        lamstring = str(self.lam).replace('.','-')

        testID += '_%s%s_batch%d_lambda%s' % (self.optimizer, lrstring, self.bSize, lamstring)

        return testID


class ExpHyperspectralPRI:
    """
    Experimental parameters for training models on HSI HyperPRI data
    """
    def __init__(self, calling_path, split_no=1, seed_num=0, comet_logging=True):
        #Define class attributes (actual paramters)
        self.now = datetime.datetime.now()

        # Basic definitions
        self.dataset = "HSI_SitS"
        self.b_size  = {'train': 2, 'val': 2, 'test': 2}
        self.device  = 'gpu'
        self.epochs  = 2000

        # Dataset Definitions - largest size is 267 MB
        self.patch_size  = (608, 968)
        self.hsi_lo      = 25
        self.hsi_hi      = 263
        self.channels    = 238   # may depend on which model is being used
        self.rescale     = 1
        self.augment     = False
        self.rotate      = False
        self.num_classes = 1
        self.label_set   = None   # Set to 'None' for normal root-soil segmentation
        self.data_dir    = f"{calling_path}/Datasets/HyperPRI"
        self.json_dir    = {
            'train': f"{self.data_dir}/data_splits/train{split_no}.json",
            'val': f"{self.data_dir}/data_splits/val{split_no}.json",
            'test': f"{self.data_dir}/data_splits/val{split_no}.json"
        }
        self.run_num    = 10 * seed_num + split_no
        self.test_transforms = None
        self.gt_test_transforms = [transforms.ToTensor()]
        if self.augment:
            self.train_transforms = [transforms.RandomCrop(self.patch_size)]
            self.gt_transforms = [transforms.RandomCrop(self.patch_size), transforms.ToTensor()]
        else:
            self.train_transforms = None  # HSI method already converts to Tensor
            self.gt_transforms = [transforms.ToTensor()]

        # Model Parameters
        self.model_name         = "CubeNET"
        self.bilinear           = False
        self.use_attention      = False
        self.use_pretrained     = False

        # DeepSpeed Testing Parameters
        self.mlp_layers         = [1650] * 10
        self.test_deepspeed     = False

        # Hyperspectral Stuff --------------------
        self.spectral_bn_size = 1650      ## Size of the bottleneck for SpectralUNET
        self.cube_featmaps    = 64        ## How many feature maps are in CubeNET's first layer

        # PyTorch required parameters
        self.criterion    = torch.nn.BCEWithLogitsLoss()
        self.optimizer    = "adam"
        self.learn_rate   = 0.001
        self.weight_decay = 0
        self.momentum     = 0.9

        # PyLightning Metrics param's -------------------------------------
        self.task         = "binary"
        self.threshold    = 0.5

        # Early Stopping parameters
        self.consecutive = None
        self.overall     = 500

        # Visualization parameters
        self.model_param_str = self.translate_load_dir()
        self.save_path = f"{calling_path}/Saved_Models/{self.dataset}/{self.model_param_str}/Run_{self.run_num}/"
        self.fig_dir   = f"{calling_path}/Saved_Models/{self.dataset}/Val_Segmentation_Maps/Run_{self.run_num}/{self.model_param_str}/"

        # Comet Logging Parameters
        self.comet_params = {
            "api_key": os.environ.get("COMET_API_KEY") if comet_logging else None,
            "workspace": os.environ.get("COMET_WORKSPACE") if comet_logging else None,
            "offline_dir": f"{calling_path}/comet_offline/",
            "project_name": "hyperpri",
            "experiment_name": f"{self.dataset}-{self.model_name}-{self.run_num}",
        }

    def change_network_param(self, new_model_name, calling_path, split_no, seed_num=0, model_params=None):
        """
        For evaluating multiple models and
        changing the parameters' model info on-the-fly
        """
        # If a dictionary of parameters is provided, overwrite the provided param's
        if model_params is not None:
            for k_idx, k in enumerate(model_params):
                this_attr = getattr(self, k, None)
                if this_attr is not None:
                    setattr(self, k, model_params[k])

        self.run_num = 10 * seed_num + split_no
        self.model_name = new_model_name
        self.model_param_str = self.translate_load_dir()
        self.save_path = f"{calling_path}/Saved_Models/{self.dataset}/{self.model_param_str}/Run_{self.run_num}/"
        self.fig_dir   = f"{calling_path}/Saved_Models/{self.dataset}/Val_Segmentation_Maps/Run_{self.run_num}/{self.model_param_str}/"

    def translate_load_dir(self):    #Generate segmentation model
        if self.model_name.lower() == 'spectralunet':
            model_str = f"{self.model_name}_{self.spectral_bn_size}"
        elif self.model_name.lower() == 'cubenet':
            model_str = f"{self.model_name}_{self.cube_featmaps}"
        #Base UNET model or UNET+ (our version of attention)
        elif self.model_name.lower() in ['unet', 'unet+']:
            model_str = self.model_name
        else:
            err_str = f"{self.model_name} is not in list of possible models\n"
            err_str += "   (accepted: UNET, UNET+, SpectralUNET, CubeNET)"
            ValueError(err_str)
        return model_str

    def get_network(self):
        if self.model_name.lower() == 'spectralunet':
            depth = self.hsi_hi - self.hsi_lo
            model = SpectralUNET(depth,
                                 self.num_classes,
                                 bn_feats=self.spectral_bn_size)

        elif self.model_name.lower() == 'cubenet':
            depth = self.hsi_hi - self.hsi_lo
            model = CubeNET(depth,
                            self.num_classes,
                            first_depth=self.cube_featmaps,
                            bilinear=self.bilinear,
                            use_attention=self.use_attention)

        else: #Show error that segmentation model is not available
            raise RuntimeError('ExpHyperspectralPRI: Invalid model')

        return model

    def get_train_data(self):
        preserve_cube = self.model_name.lower() == 'cubenet'
        data_transforms = self.train_transforms if self.train_transforms is None else transforms.Compose(self.train_transforms)
        return HyperpriDataset(root=self.data_dir,
                               img_transform=data_transforms,
                               label_transform=transforms.Compose(self.gt_transforms),
                               subset=self.label_set,
                               mode='HSI',
                               unsqueeze_img=preserve_cube,
                               hsi_lo=self.hsi_lo,
                               hsi_hi=self.hsi_hi,
                               json_file=self.json_dir.get('train', None)
        )

    def get_val_data(self):
        preserve_cube = self.model_name.lower() == 'cubenet'
        test_transform = None if self.test_transforms is None else transforms.Compose(self.test_transforms)
        return HyperpriDataset(root=self.data_dir,
                               img_transform=test_transform,
                               label_transform=transforms.Compose(self.gt_test_transforms),
                               subset=self.label_set,
                               mode='HSI',
                               unsqueeze_img=preserve_cube,
                               hsi_lo=self.hsi_lo,
                               hsi_hi=self.hsi_hi,
                               json_file=self.json_dir.get('val', None)
        )

    def get_test_data(self):
        preserve_cube = self.model_name.lower() == 'cubenet'
        test_transform = None if self.test_transforms is None else transforms.Compose(self.test_transforms)
        return HyperpriDataset(root=self.data_dir,
                               img_transform=test_transform,
                               label_transform=transforms.Compose(self.gt_test_transforms),
                               subset=self.label_set,
                               mode='HSI',
                               unsqueeze_img=preserve_cube,
                               hsi_lo=self.hsi_lo,
                               hsi_hi=self.hsi_hi,
                               json_file=self.json_dir.get('test', None)
        )

    def get_test_id(self):
        '''Creates name to identify experiment output files/folders by'''

        # Create unique identifier for test
        testID = '%d%02d%02d_%02d%02d' % (self.now.year, self.now.month, self.now.day, self.now.hour, self.now.minute)

        lrstring = str(self.learningRate).replace('.','-')
        lamstring = str(self.lam).replace('.','-')

        testID += '_%s%s_batch%d_lambda%s' % (self.optimizer, lrstring, self.bSize, lamstring)

        return testID
