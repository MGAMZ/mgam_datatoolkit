import pdb
from typing import Union

import torch
import numpy as np
from torch.nn.functional import interpolate
from scipy.spatial.distance import directed_hausdorff
from monai.metrics import compute_hausdorff_distance

from mmseg.models.losses.dice_loss import dice_loss
from mmseg.models.losses import accuracy




def AlignDimension(y_pred, y_true):
    if y_pred.ndim > y_true.ndim:
        y_pred = y_pred.argmax(dim=1)
    elif y_pred.ndim < y_true.ndim:
        y_true = y_true.argmax(dim=1)
    return y_pred, y_true



def dice_loss_array(pred: np.ndarray,
                    target: np.ndarray,
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



def accuracy_array(y_pred:np.ndarray, y_true:np.ndarray):
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



def evaluation_dice(gt_data:np.ndarray, pred_data:np.ndarray):
    gt_class = torch.from_numpy(gt_data).cuda()
    pred_class = torch.from_numpy(pred_data).cuda()
    dice = 1 - dice_loss(gt_class[None], pred_class[None], weight=None, ignore_index=None
                            ).cpu().numpy()
    return dice



def evaluation_area_metrics(gt_data:np.ndarray, pred_data:np.ndarray):
    # 计算iou, recall, precision
    gt_class = torch.from_numpy(gt_data).cuda()
    pred_class = torch.from_numpy(pred_data).cuda()
    tp = (gt_class * pred_class).sum()
    fn = gt_class.sum() - tp
    fp = pred_class.sum() - tp
    
    iou = (tp / (tp + fn + fp)).cpu().numpy()
    recall = (tp / (tp + fn)).cpu().numpy()
    precision = (tp / (tp + fp)).cpu().numpy()
    return iou, recall, precision



def evaluation_hausdorff_distance_3D(gt, pred, percentile:int=95, interpolation_ratio:Union[float|None]=0.25):
    gt = torch.from_numpy(gt).to(dtype=torch.uint8, device='cuda')
    pred = torch.from_numpy(pred).to(dtype=torch.uint8, device='cuda')
    if interpolation_ratio is not None:
        gt = interpolate(gt, scale_factor=interpolation_ratio, mode='nearest')
        pred = interpolate(pred, scale_factor=interpolation_ratio, mode='nearest')
    
    # gt, pred: [Class, D, H, W]
    # input of the calculation should be: [N, Class, D, H, W]
    value = compute_hausdorff_distance(
        y_pred = pred[None],
        y = gt[None],
        include_background = True,
        percentile = percentile,
        directed = True,
    )
    
    return value.cpu().numpy().squeeze()




if __name__ == '__main__':
    image = np.zeros((5, 128, 128))
    image[..., 32:48, 32:48] = 1
    image2 = np.roll(image, 4, axis=1)
    image2 = np.roll(image2, 4, axis=2)
    
    distance = evaluation_hausdorff_distance_3D(image, image2, None)
    print(distance)
