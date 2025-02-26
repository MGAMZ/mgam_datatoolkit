import pdb
from typing import Union

import torch
import numpy as np
from torch import Tensor
from torch.nn.functional import interpolate
from monai.metrics import compute_hausdorff_distance

from mmseg.models.losses.dice_loss import dice_loss, DiceLoss

from ..utils.DeviceSide import get_max_vram_gpu_id




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



def accuracy_tensor(y_pred:Tensor, y_true:Tensor):
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



def evaluation_hausdorff_distance_3D(gt, 
                                     pred, 
                                     percentile:int=95, 
                                     interpolation_ratio:Union[float, None]=None):
    selected_device_id = get_max_vram_gpu_id()
    gt = torch.from_numpy(gt).to(dtype=torch.uint8, device=f'cuda:{selected_device_id}')
    pred = torch.from_numpy(pred).to(dtype=torch.uint8, device=f'cuda:{selected_device_id}')
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
    ).cpu().numpy().squeeze()
    
    torch.cuda.empty_cache()
    return value


class DiceLoss_3D(DiceLoss):
    def __init__(
        self,
        ignore_1st_index: bool = False,
        batch_z: int | None = None,
        class_weight: list[float] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_1st_index = ignore_1st_index
        self.batch_z = batch_z
        self.class_weight = class_weight

    def _expand_onehot_labels_dice_3D(self, pred: Tensor, target: Tensor) -> Tensor:
        """Expand onehot labels to match the size of prediction for 3D Volumes.

        Args:
            pred (Tensor): The prediction, has a shape (N, num_class, D, H, W).
            target (Tensor): The learning label of the prediction,
                has a shape (N, D, H, W).

        Returns:
            Tensor: The target after one-hot encoding,
                has a shape (N, num_class, D, H, W).
        """
        num_classes = pred.shape[1]
        one_hot_target = torch.clamp(target, min=0, max=num_classes)
        one_hot_target = torch.nn.functional.one_hot(
            one_hot_target.to(torch.int64), num_classes + 1
        )
        one_hot_target = one_hot_target[..., :num_classes].permute(0, 4, 1, 2, 3)
        return one_hot_target

    def forward_one_patch(self, pred: Tensor, target: Tensor, *args, **kwargs):
        if pred.shape != target.shape:
            target = self._expand_onehot_labels_dice_3D(pred, target)
            assert pred.shape == target.shape
        # pred, target: [N, C, Z, Y, X]
        if self.ignore_1st_index:
            pred = pred[:, 1:, ...].contiguous()
            target = target[:, 1:, ...].contiguous()
        
        return super().forward(pred, target, *args, **kwargs)

    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        # pred: [N, C, Z, Y, X]
        assert (
            pred.shape[-3:] == target.shape[-3:]
        ), f"The [Z, Y, X] of pred {pred.shape} and target {target.shape} must be the same."

        if self.batch_z is not None:
            batch_loss = []
            
            for z in range(0, pred.shape[-3], self.batch_z):
                batch_z_loss = self.forward_one_patch(
                    pred=pred[..., z : z + self.batch_z, :, :], 
                    target=target[..., z : z + self.batch_z, :, :], 
                    *args, **kwargs
                )
                batch_loss.append(batch_z_loss)
            
            return torch.stack(batch_loss).mean()

        else:
            return self.forward_one_patch(pred, target, *args, **kwargs)



if __name__ == '__main__':
    image = np.zeros((5, 128, 128))
    image[..., 32:48, 32:48] = 1
    image2 = np.roll(image, 4, axis=1)
    image2 = np.roll(image2, 4, axis=2)
    
    distance = evaluation_hausdorff_distance_3D(image, image2, None)
    print(distance)
