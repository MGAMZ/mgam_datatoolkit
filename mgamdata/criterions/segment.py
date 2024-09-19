import pdb

import torch
import numpy as np
from nptyping import NDArray

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from mmdet.models.losses.dice_loss import dice_loss
from mmseg.models.losses import accuracy



def AlignDimension(y_pred, y_true):
    if y_pred.ndim > y_true.ndim:
        y_pred = y_pred.argmax(dim=1)
    elif y_pred.ndim < y_true.ndim:
        y_true = y_true.argmax(dim=1)
    return y_pred, y_true


def dice_loss_array(pred: NDArray,
                    target: NDArray,
                    eps=1e-3,
                    naive_dice=False,):
    assert pred.shape == target.shape
    per_class_dice = []
    
    for class_idx in np.unique(target):
        class_pred = pred==class_idx
        class_target = target==class_idx
        inputs = class_pred.reshape(class_pred.shape[0], -1)
        target = class_target.reshape(class_target.shape[0], -1)

        a = np.sum(inputs * target, 1)
        if naive_dice:
            b = np.sum(inputs, 1)
            c = np.sum(target, 1)
            d = (2 * a + eps) / (b + c + eps)
        else:
            b = np.sum(inputs * inputs, 1) + eps
            c = np.sum(target * target, 1) + eps
            d = (2 * a) / (b + c)
        
        per_class_dice.append(np.mean(1 - d))
    return np.mean(per_class_dice)


def dice_loss_tensor(y_pred:torch.Tensor, y_true:torch.Tensor):
    '''
        y_pred: [N, C, ...]
        y_true: [N, C, ...]
    '''
    y_pred, y_true = AlignDimension(y_pred, y_true)
    dice = dice_loss(y_true.flatten(1), y_pred.flatten(1))
    return dice

def accuracy_array(y_pred:NDArray, y_true:NDArray):
    '''
        y_pred: [N, ...]
        y_true: [N, ...]
    '''
    y_pred, y_true = AlignDimension(y_pred, y_true)
    correct = (y_pred == y_true).sum()
    total = np.prod(y_true.shape)
    return correct / (total + 1)


def accuracy_tensor(y_pred:torch.Tensor, y_true:torch.Tensor):
    y_pred, y_true = AlignDimension(y_pred, y_true)
    correct = (y_pred == y_true).sum().item()
    total = y_true.numel()
    return correct / total


