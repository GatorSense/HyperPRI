"""
This file is setup as a completely standalone prediction script
for the HyperPRI dataset. It only imports from an additional file
`pred_backend.py`.

It is meant for use as a single call to the following method to segment a
specified part of the HyperPRI dataset (or other similar dataset).

Current implementation only accepts HSI predictions. More logic may be
added in the future, depending on requests.

Created by request of @Ritesh313

Author: @changspencer
Date: 2024-09-19
"""
## Python dependencies
import os
import numpy as np
import matplotlib.pyplot as plt

## HSI-based dependencies - Remove spectral warnings
import spectral
spectral.settings.envi_support_nonlowercase_params = True

## PyTorch dependencies
import torch
import lightning as pl

from src.Experiments.models import UNet, SpectralUNET, CubeNET


def hyperpri_prediction(img_path: str, load_path: str, model_params: dict, model_thresh: float):
    """
    Segments the image/data-cube given at `img_path` using the `load_path` that
        corresponds to the `model_params`. The method outputs three data files:
        the image data, the thresholded prediction (using >= `model_thresh`), and
        the raw prediction.

    geq = greater than or equal to

    Arguments:
        img_path -- str path to the input data file to be segmented
        load_path -- str path to the loadable model weights (must be in LightningModule form)
        model_params -- dict that mimics model parameters in parameter files below (BareRGB, BareHSI)
        model_thresh -- float value that cuts off predictions as 0's or 1's (geq are 1's)
    Returns:
        input_data -- image data to be segmented; output for comparison with prediction mask
        thresh_pred -- Binary mask computed after using `model_thresh`
        raw_pred -- raw softmax (logistic) predictions from 0 to 1 for the `input_data`
    """
    # A bit of data validation
    m_name = model_params['model'].lower()
    img_type = model_params['dataset'].lower()
    assert img_type == 'hsi', f"Improper image type: {img_type}. Only accepting ['hsi']."
    assert m_name in ['unet', 'spectralunet', 'cubenet'], f"Invalid model name: {m_name}. Only accepting ['unet', 'spectralunet', 'cubenet']"

    print(f"Model: {model_params['model']}")
    print(f"Dataset: {model_params['dataset']}")
    print(f"Best Threshold: {model_thresh}")

    # if img_type == 'rgb':
    #     exp_params = BareRedGreenBlue()
    #     exp_params.change_network_param(model_params['model'], model_params=model_params)
    # else:
    exp_params = BareHSI()
    exp_params.change_network_param(model_params['model'], model_params=model_params)
    net = exp_params.get_network()

    # Load path must be a checkpoint for a pytorch lightning module as specified below.
    print(f"Loading from Ckpt File: {load_path}")
    pl_model = BareRootModule(exp_params)
    raw_state_dict = torch.load(load_path, map_location="cpu")['state_dict']
    pl_model.load_state_dict(raw_state_dict)
    net = pl_model.m_network

    ##### DATA LOADING AND PREPARATION - Assumes full 299 bands from ~450 - 1000 nm (2nm res) #####
    print(f"Loading and Preprocessing image: {img_path}")
    input_data = np.load(img_path)
    input_data = torch.tensor(input_data[..., 25:263])
    input_data = input_data.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    ### FEED-FORWARD INTO LOADED MODEL ###
    print(f"Predicting on an image: {img_path}")
    with torch.no_grad():
        raw_pred = net(input_data)
    raw_pred = torch.sigmoid(raw_pred).squeeze().detach()
    thresh_pred = 1 * (raw_pred >= model_thresh)

    return input_data, thresh_pred, raw_pred


class BareRGB:
    """
    Experimental parameters for training models on RGB HyperPRI data
    """
    def __init__(self):

        # Basic definitions -----------------------------------------------
        self.dataset = "RGB"
        self.device  = 'gpu'

        # Dataset Definitions ---------------------------------------------
        self.patch_size  = (608, 968)
        self.color_mode  = 'rgb'
        self.channels    = 3 if self.color_mode.lower() != 'gray' else 1
        self.num_classes = 1

        # Model Parameters ------------------------------------------------
        self.model_name         = "UNET"
        self.bilinear           = False
        self.feature_extraction = False
        self.use_attention      = False

        # PyTorch required parameters -------------------------------------
        self.criterion          = torch.nn.BCEWithLogitsLoss()

        # PyLightning Metrics param's -------------------------------------
        self.task         = "binary"

        # Visualization parameters ----------------------------------------
        self.model_param_str = self.translate_load_dir()

    def change_network_param(self, new_model_name:str, model_params:dict=None):
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

        self.model_name = new_model_name
        self.model_param_str = self.translate_load_dir()

    def translate_load_dir(self):    #Generate segmentation model
        if self.model_name.lower() in ['unet']:
            model_str = self.model_name
        else:
            err_str = f"{self.model_name} is not in list of possible models\n"
            err_str += "   (accepted: UNET)"
            raise ValueError(err_str)
        return model_str

    def get_network(self):
        if self.model_name.lower() in ['unet']:
            model = UNet(self.channels,
                         self.num_classes,
                         bilinear=self.bilinear,
                         feature_extraction=self.feature_extraction,
                         use_attention=self.use_attention)
        else: #Show error that segmentation model is not available
            raise RuntimeError('BareRedGreenBlue: Invalid model')

        return model


class BareHSI:
    """
    Experimental parameters for training models on HSI HyperPRI data
    """
    def __init__(self):

        # Basic definitions
        self.dataset = "HSI"
        self.device  = 'gpu'

        # Dataset Definitions - largest size is 267 MB
        self.patch_size  = (608, 968)  # Full size = (608, 968)
        self.hsi_lo      = 25
        self.hsi_hi      = 263
        self.channels    = 238   # may depend on which model is being used
        self.num_classes = 1

        # Model Parameters
        self.model_name         = "CubeNET"
        self.bilinear           = False
        self.use_attention      = False

        # Hyperspectral Stuff --------------------
        self.spectral_bn_size = 1650      ## Size of the bottleneck for SpectralUNET
        self.cube_featmaps    = 64        ## How many feature maps are in CubeNET's first layer

        # PyTorch required parameters
        self.criterion    = torch.nn.BCEWithLogitsLoss()

        # PyLightning Metrics param's -------------------------------------
        self.task         = "binary"

        # Visualization parameters
        self.model_param_str = self.translate_load_dir()

    def change_network_param(self, new_model_name, model_params=None):
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

        self.model_name = new_model_name
        self.model_param_str = self.translate_load_dir()

    def translate_load_dir(self):    #Generate segmentation model
        if self.model_name.lower() == 'spectralunet':
            model_str = f"{self.model_name}_{self.spectral_bn_size}"
        elif self.model_name.lower() == 'cubenet':
            model_str = f"{self.model_name}_{self.cube_featmaps}"
        #Base UNET model or UNET+ (our version of attention)
        elif self.model_name.lower() == 'unet':
            model_str = self.model_name
        else:
            err_str = f"{self.model_name} is not in list of possible models\n"
            err_str += "   (accepted: unet, spectralunet, cubenet)"
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
            raise RuntimeError('BareHyperspectral: Invalid model')

        return model


class BareRootModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.exp_params = params

        self.test_deepspeed = None

        self.p_optimizer = None
        self.p_learn_rate = None
        self.p_decay = None
        self.p_momentum = None

        self.f_criterion = params.criterion
        self.m_network = params.get_network()

        # Semantic Segmentation metrics
        self.accuracy = None
        self.pos_iou = None
        self.dice_val = None
        self.pr_curve = None
        self.save_segmaps = False
        self.threshold = 0.5

        # Testing predictions and values
        self.predict_labels = []


if __name__ == "__main__":
    #### NEEDED INPUTS ####
    rel_call_path = "C:/Users/SChan/Documents/Research Documents/GATORSENSE/sits_project/HyperPRI/"
    load_file = "Saved_Models/HSI/CubeNET_64/Run_3/Checkpoints/epoch=112-val_loss=0.067-val_dice=0.861.ckpt"
    load_path = os.path.join(rel_call_path, load_file)
    data_path = "23-03-31_77_C6_REF.npy"
    model_params = {
        'model': 'CubeNET',
        'dataset': 'HSI',
        'cube_featmaps': 64,
        'criterion': torch.nn.BCEWithLogitsLoss(),
        'segmaps': True,
    }

    # model_thresh = [
    #     0.33, 0.46, 0.39, 0.46, 0.27   # CubeNET
    # ]
    img_data, thresh_img, _ = hyperpri_prediction(data_path, load_path, model_params, 0.39)

    ### LOOK AT PREDICTION BASED ON BEST VALIDATED THRESHOLD ###
    fig, axes = plt.subplots(1, 2, dpi=120, layout='constrained')
    axes[0].imshow(img_data.squeeze().permute(1, 2, 0)[:, :, [125, 49, 0]]**(1 / 2.2))
    axes[1].imshow(thresh_img)
    axes[0].set_title("Reference (Grnd Truth) Image")
    axes[1].set_title("CubeNET Prediction")
    axes[0].axis('off')
    axes[1].axis('off')

    plt.show()