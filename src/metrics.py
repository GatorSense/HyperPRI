"""
Some functional code from https://github.com/qubvel/segmentation_models.pytorch
Modified to include Hausdorff loss from: https://github.com/SilmarilBearer/HausdorffLoss

Metrics code modified from https://github.com/GatorSense/Histological_Segmentation,
which takes modified code from various other repos.

@author: chang.spencer
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from torchmetrics.functional import average_precision
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import adjusted_rand_score as arsc
from sklearn.metrics import confusion_matrix
from scipy.ndimage.morphology import distance_transform_edt as edt
import numpy as np


class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.max(distances[indexes]))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()

        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
        ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
        ).float()

        return torch.max(right_hd, left_hd)


def _take_channels(*xs, ignore_channels=None):
    """Removes certain channels from the input's referenced
        tensor.
    Args:
        *xs: tensor 'pointer'/reference
        ignore_channels: a tuple of int's stating which channels
            should be ignored in the computation
    Returns:
        xs: tensor with filtered channels
    """
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    """Threshold input values based on the given threshold.
    Args:
        x: tensor to be thresholded
        threshold: float determining cutoff
    Returns:
        x: thresholded input tensor
    """
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou  # Initialize equivalent metric name


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def dice_coeff(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Dice coefficient between ground truth and prediction.
    This is very related to the F1 Score.
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    inter = torch.dot(pr.view(-1), gt.view(-1))
    union = torch.sum(pr) + torch.sum(gt) + eps

    dice = (2 * inter.float() + eps) / union.float()

    return dice


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: accuracy score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score

def specificity(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate specificity score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: specificity score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    tn = torch.sum((gt==0) * (pr==0))
    fp = torch.sum(pr) - tp

    score = (tn + eps) / (tn + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


def mean_average_precision(pred, true, ignore_channels=None):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        true: a tensor of shape [B, C, H, W] or [B, H, W].
        ignore_channels: a tuple of int's stating which channels
            should be ignored in the computation
    Returns:
        mAP: mean average precision of pred based on true labels
    """
    # Filter out which channels we don't want - UNTESTED
    pred, true = _take_channels(pred, true, ignore_channels=ignore_channels)

    # Figure out the total number of classes left - UNTESTED
    # print(pred.shape, true.shape)
    # print(torch.unique(true.flatten()))
    num_classes = torch.unique(true.flatten()).shape[0]

    #Compute per-class average precision
    Class_AP = average_precision(pred, true, num_classes=num_classes-1)
    
    # Convert to tensor (not sure if this has to do with # of batches or channels/classes)
    if num_classes > 2:
        Class_AP = torch.stack(Class_AP)
    
    #Return mean average precision, ignoring any NaNs
    nanmean = lambda x: torch.mean(x[x == x])
    mAP = nanmean(Class_AP)
    
    return mAP


def score_counter(true_map, pred_map, n_thresholds=50):
    '''
    Acquire the TN, FP, FN, and TP for predictions at multiple
        thresholds. For use in ROC and PR curves
    Data must be for binary labeling when input to this method.
    
    Args:
        true_map: torch tensor of the true segmentation labels
        pred_map: torch tensor of the predicted segmentation sigmoid
            probabilities
        n_thresholds: int stating the number of thresholds to test
    Returns:
        confuse_data: 
        thresholds: 
    '''
    thresholds = [i / n_thresholds for i in range(0, n_thresholds+1)]
    confuse_data = np.zeros((4, n_thresholds+1))
    for idx, t in enumerate(thresholds):
        heaviside = nn.Threshold(t, 0)

        threshold_map = torch.ceil(heaviside(pred_map))
        tn, fp, fn, tp = confusion_matrix(true_map.flatten(),
                                          threshold_map.flatten()).flatten()

        confuse_data[0, idx] = tn
        confuse_data[1, idx] = fp
        confuse_data[2, idx] = fn
        confuse_data[3, idx] = tp

    return confuse_data, thresholds


def pr_counts(fp:np.array, fn:np.array, tp:np.array):
    '''
    Given binary confusion matrix data, compute the precision and recall.
    Args:
        fp: 1-dim array of int false positive counts
        fn: 1-dim array of int false negative counts
        tp: 1-dim array of int true positive counts
    Returns:
        prec: 1-dim array of float Precision values
        rec: 1-dim array of float Recall values
    '''
    # Zero'd TP and FP input validation
    max_prec_idx = np.intersect1d(np.where(tp == 0)[0],
                                  np.where(fp == 0)[0]).min()
    prec = tp[:max_prec_idx] / (tp[:max_prec_idx] + fp[:max_prec_idx])
    rec = tp / (tp + fn)
    prec = np.concatenate((prec, [1] * (len(tp) - max_prec_idx)))

    average_prec = 0
    for rec_idx in range(1, len(rec)):
        average_prec += (rec[rec_idx - 1] - rec[rec_idx]) * prec[rec_idx - 1]

    return prec, rec, average_prec


def Average_Metric(input,target,pos_wt=None,metric_name='Prec',ignore_channels=None):
    """Metrics for batches
        Capable of computing the following metrics for any number of labels.
    """
    s = 0
    haus_count = 0
    hausdorff_pytorch = HausdorffDistance()
    
    for i, c in enumerate(zip(input, target)):
        if metric_name == 'Precision':
            s +=  precision(c[1],c[0],ignore_channels=ignore_channels).item()
        elif metric_name == 'Recall':
            s += recall(c[1],c[0],ignore_channels=ignore_channels).item()
        elif metric_name == 'F1':
            s += f_score(c[1],c[0],ignore_channels=ignore_channels).item()
        elif metric_name == "mAP":
            s += mean_average_precision(c[1], c[0], ignore_channels=ignore_channels).item()
        elif metric_name == 'Hausdorff':
            temp_haus = hausdorff_pytorch.compute(c[1].unsqueeze(0),c[0].unsqueeze(0)).item()
            if temp_haus == np.inf: #If output does not have positive class, do not include in avg (GT has few positive ROI) 
                haus_count +=1
            else:
                s += temp_haus
        elif metric_name == 'Jaccard':
            s += iou(c[1],c[0],ignore_channels=ignore_channels).item()
        elif metric_name == 'Rand':
            s += arsc(c[1].cpu().numpy().reshape(-1).astype(int),
                      c[0].cpu().numpy().reshape(-1).astype(int))
        elif metric_name == 'IOU_All': #Background
            s += jsc(c[1].cpu().numpy().reshape(-1).astype(int),
                      c[0].cpu().numpy().reshape(-1).astype(int),average='macro')
        elif metric_name == 'Acc':
            s += 100*np.sum(c[1].cpu().numpy().reshape(-1).astype(int) == 
                            c[0].cpu().numpy().reshape(-1).astype(int))/(len(c[0].cpu().numpy().reshape(-1).astype(int)))
        elif metric_name == 'BCE':
            s += F.binary_cross_entropy_with_logits(c[0],c[1],
                                                    pos_weight=pos_wt).item()
        elif metric_name == 'Dice_Loss':
            s += dice_coeff(c[1],c[0],ignore_channels=ignore_channels).item()
        elif metric_name == 'Spec':
            s += specificity(c[1],c[0],ignore_channels=ignore_channels).item()
        else:
            raise RuntimeError('Metric is not implemented')
        
    if metric_name == 'Hausdorff':
        return s, haus_count
    else:
        # print(s)
        return s
