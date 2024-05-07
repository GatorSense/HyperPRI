import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, BasePredictionWriter, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, CometLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities import grad_norm

if sys.platform != "win32":
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, convert_zero_checkpoint_to_fp32_state_dict
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from lightning.pytorch.strategies import FSDPStrategy

from torchmetrics import Accuracy, JaccardIndex, Dice, PrecisionRecallCurve, AveragePrecision
from torchmetrics.classification import BinaryConfusionMatrix

from .Experiments.models import *


# From PyTorch Performance tuning (should make faster)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class OutputWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        pass


class RootLightningModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.exp_params = params

        self.test_deepspeed = params.test_deepspeed

        self.p_optimizer = params.optimizer
        self.p_learn_rate = params.learn_rate
        self.p_decay = params.weight_decay
        self.p_momentum = params.momentum

        self.f_criterion = params.criterion
        self.m_network = params.get_network()

        # Semantic Segmentation metrics
        self.accuracy = Accuracy(task=params.task, num_classes=params.num_classes)
        self.pos_iou = JaccardIndex(task=params.task, num_classes=params.num_classes, threshold=params.threshold)
        self.dice_val = Dice(num_classes=params.num_classes+1,  # Needs soil pixels
                             threshold=params.threshold,
                             zero_division=1e-12,
                             ignore_index=0 if params.task == 'binary' else None)
        self.pr_curve = PrecisionRecallCurve(params.task)  # For validation
        self.save_segmaps = False
        self.threshold = 0.5

        # Auto-saves/-prints the models parameters
        if not params.augment:
            self.save_hyperparameters()   # Can't rely on this for different types of Torch transform lists

        # Testing predictions and values
        self.predict_labels = []

    def training_step(self, batch, batch_idx):
        mask = batch['mask'].to(torch.int32)

        if hasattr(self.m_network, "analyze") and self.m_network.analyze:
            pred, _ = self.m_network(batch['image'])
        else:
            pred = self.m_network(batch['image'])
        loss = self.f_criterion(pred, batch['mask'])

        seg = torch.sigmoid(pred.detach()) > self.threshold
        acc = self.accuracy(seg, mask)
        dice = self.dice_val(seg, mask)
        pos_iou = self.pos_iou(seg, mask)

        self.log('tr_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('tr_acc', acc, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
        self.log('tr_dice', dice, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log('tr_pos_iou', pos_iou, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        mask = batch['mask'].to(torch.int32)

        if hasattr(self.m_network, "analyze") and self.m_network.analyze:
            pred, _ = self.m_network(batch['image'])
        else:
            pred = self.m_network(batch['image'])

        loss = self.f_criterion(pred, batch['mask'])

        seg = torch.sigmoid(pred.detach()) > 0.5
        acc = self.accuracy(seg, mask)
        dice = self.dice_val(seg, mask)
        pos_iou = self.pos_iou(seg, mask)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
        self.log('val_dice', dice, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val_pos_iou', pos_iou, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        self.predict_labels.append(batch['mask'].cpu())
        if hasattr(self.m_network, "analyze") and self.m_network.analyze:
            pred, _ = self.m_network(batch['image'])
            pred = pred.cpu()
        else:
            pred = self.m_network(batch['image']).cpu()

        if self.save_segmaps:
            masks = batch['mask'].cpu()
            filenames = batch['index']
            print(self.threshold)
            eval_color_segmaps(batch['image'].cpu(),
                               filenames,
                               pred,
                               masks,
                               self.exp_params,
                               threshold=self.threshold)

        return pred

    def configure_optimizers(self):

        # Get all network modules
        params_dict_list = self.m_network.parameters()

        if self.test_deepspeed is not None and self.test_deepspeed:
            optimizer = DeepSpeedCPUAdam(self.trainer.model.parameters())
        if self.p_optimizer.upper() == 'ADAM':
            optimizer = optim.Adam(params_dict_list,
                                   lr=self.p_learn_rate,
                                   weight_decay=self.p_decay)
        elif self.p_optimizer.upper() == 'SGD':
            optimizer = optim.SGD(params_dict_list,
                                  lr=self.p_learn_rate,
                                  momentum=self.p_momentum,
                                  weight_decay=self.p_decay)
        else:
            raise ValueError(f'Unknown Optimizer name: {self.p_optimizer.upper()}')

        return optimizer


def consolidate_deepspeed_two(ckpt_path):
    """
    Consolidate the deepspeed (zero stage 2) model and optimizer
        states into a single checkpoint.
    """
    if sys.platform != "win32":
        # CONVERT DSZeRO-2 CKPT to state dict...
        #! Workaround for potentially regex-significant dir names
        print(f"      (Converting DeepSpeed ZeRO-2 Checkpoint...)")
        cwd = os.path.realpath(os.path.curdir)
        os.chdir(ckpt_path)
        temp_path = "./"
        convert_zero_checkpoint_to_fp32_state_dict(temp_path, "deepspeed.ckpt")
        os.chdir(cwd)

        dszero_state_dict = torch.load(os.path.join(ckpt_path, "deepspeed.ckpt"), map_location='cpu')
        state_dict = {}
        for key_wgt in dszero_state_dict:
            new_key = key_wgt.replace("_forward_module.m_network.", "")

            if "feat_ext" in new_key:
                #! Curious that this didn't get saved during training;
                #!     perhaps because it wasn't called in the forward loop.
                continue

            state_dict[new_key] = dszero_state_dict[key_wgt]
    else:
        state_dict = None
        RuntimeError("Cannot convert ZeRO-2 checkpoint on Windows; please convert it to a .ckpt file first using a Linux-based system.")

    return state_dict


def eval_color_segmaps(batch_img, batch_name, batch_pred, batch_mask, exp_params, threshold=0.5):

    fig_dir = os.path.join(exp_params.fig_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    print()
    for idx, img in enumerate(batch_img):
        pred = batch_pred[idx]
        true_mask = batch_mask[idx].squeeze()
        img_name = batch_name[idx]

        if len(img.shape) > 3:
            img = img.squeeze()

        print(f"Saving {img_name}...")
        print(f"   Image Shape: {img.shape}")
        if exp_params.dataset.lower() == "hsi_sits":
            # Assuming hsi indices [25, 263] == [~450nm, ~926nm]
            hsi_rgb = [125, 49, 0]  # R - 700nm, G - 546nm, B - 436nm
            color_correction = img[hsi_rgb, :, :]**(1 / 2.2)  # Gamma correction
            img = color_correction

        # Single prediction image
        pred = (torch.sigmoid(pred) > threshold).float()

        #Plot masks only
        M, N = pred.shape[-2], pred.shape[-1]
        temp_overlap = np.zeros((M, N, 3))
        preds_mask = pred.permute(1,2,0)[:,:,0].numpy().astype(dtype=bool)
        gt_mask = true_mask.numpy().astype(dtype=bool)
        temp_overlap[:,:,0] = preds_mask
        temp_overlap[:,:,1] = gt_mask

        #Convert to color blind
        #Output
        temp_overlap[preds_mask,:] = [202/255, 0/255, 32/255]  # Red
        temp_overlap[gt_mask, :] = [5/255, 133/255, 176/255]   # Blue
        agreement = preds_mask * gt_mask
        temp_overlap[agreement, :] = [155/255, 191/255, 133/255]  # Green

        # Save the color-coded prediction as a separate image
        model_fig = plt.figure(dpi=200)
        plt.title(f"{exp_params.model_param_str} - {img_name}")
        plt.imshow(img.permute(1,2,0))
        plt.imshow(temp_overlap, alpha=0.6)
        plt.tick_params(axis='both', labelsize=0, length = 0)
        model_fig.savefig(f"{fig_dir}/{img_name}_seg.png", dpi=200, bbox_inches='tight')
        plt.close(model_fig)


def load_val_model(params):
    """
    For loading a pytorch lightning model from previously saved weights.
    """
    #* Cross-branch compatibility to PyTorch CKPTs (branch `sits_dev`)
    if os.path.exists(os.path.join(params.save_path, 'Checkpoints')):
        # Choose the most recent, best model in the diceCheckpoints path.
        load_path = os.path.join(params.save_path, 'Checkpoints')
        ckpts = os.listdir(load_path)
        best_ckpt = "last.ckpt"
        if len(ckpts) >= 2:
            # ckpts.remove("last.ckpt")
            latest_time = 0
            for c in ckpts:
                if 'last' in c:
                    continue
                temp_path = os.path.join(load_path, c)
                if os.path.getmtime(temp_path) > latest_time:
                    best_ckpt = c
                    latest_time = os.path.getmtime(temp_path)

        ckpt_path = os.path.join(params.save_path, 'Checkpoints', best_ckpt)
    else:
        ckpt_path = os.path.join(params.save_path, "best_wts.pt")
    print(f"   LOADING FROM CKPT FILE: {ckpt_path}")

    pl_model = RootLightningModel(params)
    if os.path.isdir(ckpt_path):  # Likely a deepspeed checkpoint. Load using the given folder
        # # CONVERT DSZeRO-2 CKPT to state dict...
        if sys.platform != "win32":
            state_dict = consolidate_deepspeed_two(ckpt_path)
        else:
            state_dict = None
            RuntimeError("Cannot load DeepSpeed checkpoint on Windows; please convert it to a .ckpt file first using a Linux-based system.")

        if hasattr(pl_model.m_network, "feat_ext"):
            del pl_model.m_network.feat_ext
        pl_model.m_network.load_state_dict(state_dict)
    else:
        raw_state_dict = torch.load(ckpt_path, map_location="cpu")
        py_lightning = False
        if "pytorch-lightning_version" in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['state_dict']
            py_lightning = True   # Signifies training by Pytorch lightning Trainer?

        state_dict = {}
        if not py_lightning:
            for k in raw_state_dict.keys():
                #? These keys have always been so difficult to match up...
                if "module." in k:
                    my_key = k.replace("module.", "m_network.", 1)
                else:
                    my_key = "m_network." + k

                state_dict[my_key] = raw_state_dict[k]
        else:
            state_dict = raw_state_dict

        pl_model.load_state_dict(state_dict)

    return pl_model


def train_net(params, checkpoint=None, model_parallel:bool=False):
    """
    Description go here
    """
    # Create data loader objects
    if sys.platform == 'win32':
        trainLoader = DataLoader(params.get_train_data(), batch_size=params.b_size['train'], shuffle=True)
        valLoader = DataLoader(params.get_val_data(), batch_size=params.b_size['val'], shuffle=False)
    else:
        trainLoader = DataLoader(params.get_train_data(), batch_size=params.b_size['train'], shuffle=True, num_workers=0)
        valLoader = DataLoader(params.get_val_data(), batch_size=params.b_size['val'], shuffle=False, num_workers=0)

    # Create early stopper pytorch lightning callback
    ES = EarlyStopping('val_loss', patience=params.overall, check_on_train_epoch_end=True, verbose=True)

    # Create callback to save top 10 best models from training
    MCK = ModelCheckpoint(monitor='val_loss', mode='min', save_last=True, save_top_k=1, save_weights_only=False,
                        dirpath=os.path.join(params.save_path, 'Checkpoints'),
                        filename='{epoch}-{val_loss:.3f}-{val_dice:.3f}')
    MCK2 = ModelCheckpoint(monitor='val_dice', mode='max', save_last=True, save_top_k=1, save_weights_only=True,
                        dirpath=os.path.join(params.save_path, 'diceCheckpoints'),
                        filename='{epoch}-{val_loss:.3f}-{val_dice:.3f}')

    # Create callback for tracking learning rate
    LR = LearningRateMonitor(logging_interval='step')

    pl_model = RootLightningModel(params)

    #* Log all hyperparameters
    csvlogger = CSVLogger(os.path.join(params.save_path, 'LOGS'))
    tblogger = TensorBoardLogger(os.path.join(params.save_path, 'LOGS'))
    logger_list = [csvlogger, tblogger]

    if params.comet_params:  # Connect with online logging module
        comet_logger = CometLogger(api_key=params.comet_params['api_key'],
                                   save_dir=params.comet_params['offline_dir'],
                                   workspace=params.comet_params['workspace'],
                                   project_name=params.comet_params['project_name'],
                                   experiment_name=params.comet_params['experiment_name'])
        logger_list.append(comet_logger)

        comet_logger.log_hyperparams(params.__dict__)
        comet_logger.log_graph(pl_model)
    # tblogger.log_hyperparams(params.__dict__)  # Unable to log different types of torch transform lists

    # Load a checkpoint from abruptly ended experiments
    if checkpoint:
        print("Searching available checkpoints...")
        load_path = os.path.join(params.save_path, "Checkpoints")

        if not os.path.exists(load_path):  # Could not find a checkpoint to be loaded
            print("Split does not have a checkpoint to load...Starting from beginning")
            ckpt_files = None
        else:
            ckpts = os.listdir(load_path)
            best_ckpt = ckpts[0]
            if len(ckpts) >= 2:
                latest_time = 0
                for c in ckpts:
                    # Only take the last checkpoint saved...
                    print(f"   {c}")
                    if "last" not in c:
                        continue

                    temp_path = os.path.join(load_path, c)
                    if os.path.getmtime(temp_path) > latest_time:
                        print(f"      Most recent...")
                        best_ckpt = c
                        latest_time = os.path.getmtime(temp_path)
            ckpt_files = os.path.join(load_path, best_ckpt)
            print(f"Loading from {ckpt_files}")

    else:
        ckpt_files = None

    # Do the actual training/fitting of model
    if model_parallel and params.device == "gpu":

        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")

        if params.test_deepspeed is not None and params.test_deepspeed:
            strat = DeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=False,
            )
        else:
            strat = "deepspeed_stage_2"

        pl_trainer = pl.Trainer(max_epochs=params.epochs,
                                accelerator="gpu",
                                num_nodes=1,
                                devices=-1,
                                strategy=strat,
                                precision='bf16-mixed',   #! Negative of model-parallel methods
                                deterministic='warn',
                                logger=logger_list,
                                callbacks=[ES, MCK, LR, MCK2],
                                log_every_n_steps=1)
    elif params.device == "gpu" and torch.cuda.device_count() > 1:
        pl_trainer = pl.Trainer(max_epochs=params.epochs,
                                accelerator=params.device,
                                devices=-1,
                                strategy="ddp",
                                deterministic='warn',
                                logger=logger_list,
                                callbacks=[ES, MCK, LR, MCK2],
                                log_every_n_steps=1)
    else:
        pl_trainer = pl.Trainer(max_epochs=params.epochs,
                                accelerator=params.device,
                                # devices=1,
                                deterministic='warn',
                                logger=logger_list,
                                callbacks=[ES, MCK, LR, MCK2],
                                log_every_n_steps=1,)
                                # limit_train_batches=1,
                                # limit_val_batches=1)
        # print("\n!!!! LIMITING NUMBER OF TRAIN/VAL BATCHES !!!!")

    pl_trainer.fit(model=pl_model,
                   train_dataloaders=trainLoader,
                   val_dataloaders=valLoader,
                   ckpt_path=ckpt_files)

    return pl_trainer


def validate_net(val_data, params, pl_trainer:pl.Trainer=None, save_segmaps=False):
    """
    Description go here
    """
    if sys.platform == 'win32':
        val_loader = DataLoader(val_data, batch_size=params.b_size['test'], shuffle=False)
    else:
        val_loader = DataLoader(val_data, batch_size=params.b_size['test'], shuffle=False,
                                num_workers=2, persistent_workers=True)

    #* Cross-branch compatibility to PyTorch CKPTs (branch `sits_dev`)
    if os.path.exists(os.path.join(params.save_path, 'Checkpoints')):
        # Choose the most recent, best model in the diceCheckpoints path.
        load_path = os.path.join(params.save_path, 'Checkpoints')
        ckpts = os.listdir(load_path)
        best_ckpt = "last.ckpt"
        if len(ckpts) >= 2:
            latest_time = 0
            for c in ckpts:
                if 'last' in c:
                    continue
                temp_path = os.path.join(load_path, c)
                if os.path.getmtime(temp_path) > latest_time:
                    best_ckpt = c
                    latest_time = os.path.getmtime(temp_path)

        ckpt_path = os.path.join(params.save_path, 'Checkpoints', best_ckpt)
    else:
        ckpt_path = os.path.join(params.save_path, "best_wts.pt")
    print(f"   LOADING FROM CKPT FILE: {ckpt_path}")

    pl_model = RootLightningModel(params)
    if os.path.isdir(ckpt_path):  # Likely a deepspeed checkpoint. Load using the given folder
        print("!!!! LOADING DEEPSPEED !!!!")
        # # CONVERT DSZeRO-2 CKPT to state dict...
        if sys.platform != "win32":
            state_dict = consolidate_deepspeed_two(ckpt_path)
        else:
            state_dict = None
            RuntimeError("Cannot load DeepSpeed checkpoint on Windows; please convert it to a .ckpt file first using a Linux-based system.")

        if hasattr(pl_model.m_network, "feat_ext"):
            del pl_model.m_network.feat_ext
        pl_model.m_network.load_state_dict(state_dict)
    else:
        raw_state_dict = torch.load(ckpt_path, map_location="cpu")
        py_lightning = False
        if "pytorch-lightning_version" in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['state_dict']
            py_lightning = True   # Signifies training by Pytorch lightning Trainer?

        state_dict = {}
        if not py_lightning:
            for k in raw_state_dict.keys():
                #? These keys have always been so difficult to match up...
                if "module." in k:
                    my_key = k.replace("module.", "m_network.", 1)
                else:
                    my_key = "m_network." + k

                state_dict[my_key] = raw_state_dict[k]
        else:
            state_dict = raw_state_dict

        pl_model.load_state_dict(state_dict)

    if pl_trainer is None:
        pl_trainer = pl.Trainer(accelerator=params.device, devices=1, inference_mode=True)
    # Assumed to be after a training cycle
    preds = pl_trainer.predict(pl_model, val_loader, return_predictions=True)

    # Produce the plots...
    model_preds = torch.sigmoid(torch.cat(preds, dim=0).flatten())
    batch_masks = torch.cat(pl_model.predict_labels, dim=0).flatten().to(torch.long)

    print("   COMPUTING PRECISION-RECALL CURVE...")
    my_pr_curve = PrecisionRecallCurve('binary', thresholds=500)
    curve_info = my_pr_curve(model_preds, batch_masks)

    # Excluding top and bottom thresholds for best DICE choice...
    print(f"Ele's in PR Curve: {len(curve_info[0])}")
    pr_crop = int(len(curve_info[0]) // 100)
    temp_prec = curve_info[0][pr_crop:-pr_crop]
    temp_rec = curve_info[1][pr_crop:-pr_crop]
    temp_thresh = curve_info[2][pr_crop:-pr_crop]
    dice_info = (2 * temp_prec * temp_rec / (temp_prec + temp_rec))
    best_dice_idx = torch.argmax(dice_info)
    best_threshold = torch.round(temp_thresh[best_dice_idx].to(torch.float), decimals=2)
    curve_prec = temp_prec[best_dice_idx]
    curve_recall = temp_rec[best_dice_idx]

    print(f"\n{params.model_name}\n   Best Threshold {best_threshold:.3f}:")
    avg_prec = AveragePrecision(task='binary')
    acc_calc = Accuracy(task='binary', num_classes=params.num_classes)
    iou_calc = JaccardIndex(task='binary', num_classes=params.num_classes,
                            threshold=best_threshold.item())
    confmat_calc = BinaryConfusionMatrix(threshold=best_threshold.item())

    binary_seg = 1. * (model_preds > best_threshold)
    best_acc = acc_calc(binary_seg, batch_masks)
    print(f"      Pixel Acc: {best_acc:.3f}")
    print(f"      Precision: {curve_prec:.3f}")
    print(f"      Recall   : {curve_recall:.3f}")

    best_dice = 2 * curve_prec * curve_recall / (curve_prec + curve_recall)
    print(f"      DICE     : {best_dice:.3f}")

    best_iou = iou_calc(model_preds, batch_masks)
    print(f"      +IOU     : {best_iou:.3f}")

    prec_info = avg_prec(model_preds, batch_masks)
    print(f"      Avg Prec : {prec_info:.3f}\n")

    conf_info = confmat_calc(model_preds, batch_masks)
    conf_info = conf_info / conf_info.sum(dim=-1, keepdim=True)
    print(f"      Conf Mat : {conf_info[0].numpy().tolist()}")
    print(f"                 {conf_info[1].numpy().tolist()}")

    plt.figure(dpi=120)
    plt.plot(curve_info[1], curve_info[0], label=f'AP = {prec_info:.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(params.save_path, "pr_curve.png"))
    plt.close()

    # Change how undefined values in torchmetrics are dealt with via interpolation.
    # - At the last/highest threshold of 1 (sometimes no TP or FP samples for precision)
    # - This mostly applies to SpectralUNET's curves.
    print(f"final values -----> {curve_info[0][-5:]}")
    if curve_info[0][-2] < 1e-6:
        print("    Modifying...")
        curve_info[0][-2] = (1 + curve_info[0][-3]) / 2

    # have the option of saving segmentation masks for particular models
    if save_segmaps:
        pl_model.save_segmaps = True
        pl_model.threshold = best_threshold
        pl_trainer.predict(pl_model, val_loader, return_predictions=False)

    # Return precision, recall, and threshold info...
    return curve_info


def test_net(test_data, params, best_threshold, pl_trainer=None, save_segmaps=False):
    """
    Description go here
    """
    if sys.platform == 'win32':
        test_loader = DataLoader(test_data, batch_size=params.b_size['test'], shuffle=False)
    else:
        test_loader = DataLoader(test_data, batch_size=params.b_size['test'], shuffle=False,
                                 num_workers=2, persistent_workers=True)
    pl_model = load_val_model(params)
    pl_model.save_segmaps = save_segmaps
    pl_model.threshold = best_threshold

    if pl_trainer is None:
        pl_trainer = pl.Trainer(accelerator=params.device, devices=1, inference_mode=True)

    # Assumed to be after a training cycle
    preds = pl_trainer.predict(pl_model, test_loader, return_predictions=True)

    # Produce the plots...
    model_preds = torch.sigmoid(torch.cat(preds, dim=0))
    batch_masks = torch.cat(pl_model.predict_labels, dim=0).flatten().to(torch.long)

    print(f"Threshold {best_threshold:.3f}:")
    avg_prec = AveragePrecision(task='binary')
    dice_calc = Dice(num_classes=params.num_classes,  # Needs soil pixels
                     threshold=best_threshold,
                     zero_division=1e-12)
    acc_calc = Accuracy(task='binary', num_classes=params.num_classes)
    iou_calc = JaccardIndex(task='binary', num_classes=params.num_classes,
                            threshold=best_threshold)
    confmat_calc = BinaryConfusionMatrix(threshold=best_threshold)

    binary_seg = (1. * (model_preds > best_threshold)).flatten()
    best_acc = acc_calc(binary_seg, batch_masks)
    print(f"      Pixel Acc: {best_acc:.3f}")

    best_dice = dice_calc(binary_seg, batch_masks)
    print(f"      DICE     : {best_dice:.3f}")

    best_iou = iou_calc(binary_seg, batch_masks)
    print(f"      +IOU     : {best_iou:.3f}")

    prec_info = avg_prec(model_preds.flatten(), batch_masks)
    print(f"      Avg Prec : {prec_info:.3f}\n")

    conf_info = confmat_calc(binary_seg, batch_masks)
    conf_info = conf_info / conf_info.sum(dim=-1, keepdim=True)
    print(f"      Conf Mat : {conf_info[0].numpy().tolist()}")
    print(f"                 {conf_info[1].numpy().tolist()}")
