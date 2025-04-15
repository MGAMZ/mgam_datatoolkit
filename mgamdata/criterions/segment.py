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


class DiceLoss_3D(torch.nn.Module):
    def __init__(
        self,
        ignore_1st_index: bool = False, # Keep: Option to exclude background class
        eps=1e-5,
        ignore_index:int|None=None, # Keep: Standard practice to ignore certain classes
        loss_name="loss_dice",
        use_softmax: bool = True, # Add option to apply softmax internally
        squared_pred: bool = True, # Use squared prediction/target in denominator (common variant)
    ):
        """
        Standard 3D Dice Loss.

        Args:
            ignore_1st_index (bool): If True, exclude the first class (index 0, usually background)
                                     from the loss calculation. Default: False.
            eps (float): Smoothing factor to avoid division by zero. Default: 1e-5.
            ignore_index (int | None): Specifies a class index to ignore. If set, this class's contribution
                                       to the loss will be excluded during averaging. Default: None.
            loss_name (str): Name for the loss instance. Default: "loss_dice".
            use_softmax (bool): Apply softmax to the prediction tensor before calculating loss.
                                Assumes prediction is logits if True. Default: True.
            squared_pred (bool): Whether to square the prediction and target tensors in the denominator.
                                 Default: True.
        """
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.loss_name = self._loss_name = loss_name
        self.ignore_1st_index = ignore_1st_index
        self.use_softmax = use_softmax
        self.squared_pred = squared_pred

    def _expand_onehot_labels_dice_3D(self, pred: Tensor, target: Tensor) -> Tensor:
        """Expand onehot labels to match the size of prediction for 3D Volumes."""
        num_classes = pred.shape[1]
        # Clamp target to handle potential out-of-bounds indices before one-hot encoding
        clamped_target = torch.clamp(target, min=0, max=num_classes - 1)

        one_hot_target = torch.nn.functional.one_hot(
            clamped_target.to(torch.int64),
            num_classes=num_classes
        )
        # Permute to put class dimension second: [N, Z, Y, X, C] -> [N, C, Z, Y, X]
        one_hot_target = one_hot_target.permute(0, 4, 1, 2, 3)
        return one_hot_target

    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        """
        Calculates the standard Dice Loss.

        Args:
            pred (Tensor): The prediction tensor (logits or probabilities).
                           Shape: (N, C, Z, Y, X).
            target (Tensor): The ground truth tensor.
                             Shape: (N, Z, Y, X) [Integer class indices]
                             or (N, 1, Z, Y, X) [Integer class indices]
                             or (N, C, Z, Y, X) [One-hot encoded].

        Returns:
            Tensor: The calculated Dice loss (scalar).
        """
        # Apply Softmax if requested (assumes pred are logits)
        if self.use_softmax:
            pred = torch.softmax(pred, dim=1)

        # --- Input Validation and Preparation ---
        pred_spatial_shape = pred.shape[2:]
        target_spatial_shape = target.shape[-3:]
        if pred_spatial_shape != target_spatial_shape:
             raise ValueError(f"Spatial dimensions of pred {pred.shape} and target {target.shape} must match.")

        num_classes = pred.shape[1]

        # Convert target to one-hot if it's not already
        if target.ndim == pred.ndim - 1: # Shape (N, Z, Y, X)
            target = target.unsqueeze(1) # Add channel dim: (N, 1, Z, Y, X)

        if target.shape[1] == 1 and pred.shape[1] > 1: # Shape (N, 1, Z, Y, X) with class indices
             target_one_hot = self._expand_onehot_labels_dice_3D(pred, target.squeeze(1))
        elif target.shape[1] == num_classes: # Already one-hot
             target_one_hot = target
        else:
             raise ValueError(f"Target shape {target.shape} is not compatible with pred shape {pred.shape}")

        assert pred.shape == target_one_hot.shape, "Internal error: pred and target_one_hot shapes mismatch."

        N, C = pred.shape[:2]

        # --- Class Masking ---
        # Determine which classes to include in the loss calculation
        class_mask = torch.ones(C, dtype=torch.bool, device=pred.device)
        if self.ignore_1st_index:
            if C == 0: return torch.tensor(0.0, device=pred.device, requires_grad=True) # No classes left
            class_mask[0] = False
        if self.ignore_index is not None:
            if 0 <= self.ignore_index < C:
                class_mask[self.ignore_index] = False
            # else: Optional: Warn if ignore_index is out of bounds

        # Select the relevant slices of pred and target based on the mask
        pred_masked = pred[:, class_mask, ...]
        target_masked = target_one_hot[:, class_mask, ...]

        # Check if any classes remain after masking
        num_valid_classes = pred_masked.shape[1]
        if num_valid_classes == 0:
            # print("Warning: No classes left after applying ignore_1st_index and ignore_index.")
            return torch.tensor(0.0, device=pred.device, requires_grad=True) # Return 0 loss if no classes are considered

        # --- Dice Calculation ---
        # Reduce over spatial dimensions (Z, Y, X)
        dims_to_reduce = tuple(range(2, pred_masked.ndim))

        intersection = torch.sum(pred_masked * target_masked, dim=dims_to_reduce) # Shape: [N, num_valid_classes]

        if self.squared_pred:
            pred_sum = torch.sum(pred_masked * pred_masked, dim=dims_to_reduce)
            target_sum = torch.sum(target_masked * target_masked, dim=dims_to_reduce)
        else: # Use sum of probabilities/binary values
            pred_sum = torch.sum(pred_masked, dim=dims_to_reduce)
            target_sum = torch.sum(target_masked, dim=dims_to_reduce)

        numerator = 2.0 * intersection
        denominator = pred_sum + target_sum + self.eps

        dice_coeff_per_class = numerator / denominator # Shape: [N, num_valid_classes]
        loss_per_class = 1.0 - dice_coeff_per_class # Shape: [N, num_valid_classes]

        # --- NOTE Averaging ---
        # Average loss across the valid classes for each sample
        # Handle cases where a class might have zero ground truth and zero prediction correctly (dice_coeff=1, loss=0)
        # The epsilon handles division by zero, but ensure logic is sound.
        loss_per_sample = loss_per_class.mean(dim=1) # Average over valid classes, Shape: [N]=
        # Average loss across the batch
        final_loss = loss_per_sample.mean() # Average over batch, Shape: scalar

        return final_loss


class CrossEntropyLoss_3D(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        ignore_1st_index: bool = False,
        batch_z: int | None = None,
        class_weight: list[float] | None = None,
        loss_weight: float = 1.0,
        loss_name: str = "loss_CrossEntropyLoss3D",
        *args, **kwargs,
    ):
        # 如果提供了class_weight，将其转换为tensor并传递给父类
        if class_weight is not None:
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
        super().__init__(weight=class_weight, *args, **kwargs)
        self.ignore_1st_index = ignore_1st_index
        self.batch_z = batch_z
        self.loss_weight = loss_weight
        self.loss_name = loss_name
    
    def forward_one_patch(self, 
                          pred: Tensor, 
                          target: Tensor, 
                          weight:float|None=None, 
                          *args, **kwargs):
        
        target = target.long()
        # 检查target是否需要转换为类别索引
        if len(target.shape) == len(pred.shape):
            target = target.argmax(dim=1)
        
        # 如果需要忽略第一个索引
        if self.ignore_1st_index:
            # 去除预测中的第一个通道
            pred = pred[:, 1:, ...].contiguous()
            # 调整目标的类别索引
            mask = target > 0
            target = target - mask.long()
        
        # torch.nn.CrossEntropyLoss
        loss = super().forward(pred, target)
        
        if self.loss_weight != 1.0:
            loss = self.loss_weight * loss
        if weight is not None:
            loss *= weight
        
        return loss
    
    def forward(self, 
                pred: Tensor, 
                target: Tensor, 
                weight:float|None=None, 
                ignore_index:list[int]|None=None, 
                *args, **kwargs):
        # pred: [N, C, Z, Y, X]
        # 检查空间维度是否匹配
        pred_spatial_shape = pred.shape[-3:]
        target_spatial_shape = target.shape[-3:]
        
        if len(target.shape) == len(pred.shape):
            # target是one-hot编码
            assert pred.shape == target.shape, \
                f"For one-hot encoded target, shapes of pred {pred.shape} and target {target.shape} must match exactly."
        else:
            # target是类别索引
            assert pred_spatial_shape == target_spatial_shape, \
                f"The spatial dimensions [Z, Y, X] of pred {pred.shape} and target {target.shape} must match."
            
        if self.batch_z is not None:
            batch_loss = []
            
            for z in range(0, pred.shape[-3], self.batch_z):
                z_end = min(z + self.batch_z, pred.shape[-3])
                batch_z_loss = self.forward_one_patch(
                    pred=pred[..., z:z_end, :, :], 
                    target=target[..., z:z_end, :, :], 
                    weight=weight,
                    *args, **kwargs
                )
                batch_loss.append(batch_z_loss)
            
            return torch.stack(batch_loss).mean()
        else:
            return self.forward_one_patch(pred, target, *args, **kwargs)
