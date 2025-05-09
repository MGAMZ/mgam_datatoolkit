import pdb

import torch
import numpy as np
from torch import Tensor
from torch.nn.functional import interpolate



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
    from mmseg.models.losses.dice_loss import dice_loss
    gt_class = torch.from_numpy(gt_data).cuda()
    pred_class = torch.from_numpy(pred_data).cuda()
    dice = 1 - dice_loss(gt_class[None],
                         pred_class[None],
                         weight=None, 
                         ignore_index=None).cpu().numpy()
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
                                     interpolation_ratio:float|None=None):
    from monai.metrics import compute_hausdorff_distance
    from ..utils.DeviceSide import get_max_vram_gpu_id
    
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


class SplitZ_Loss(torch.nn.Module):
    def __init__(self, split_Z:bool = False, **kwargs):
        super().__init__()
        self.split_Z = split_Z
    
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        """
        Args:
            pred: logits with shape [B, C, Z, Y, X]
            target: shape [B, C, Z, Y, X] or [B, Z, Y, X]
        """
        
        # Input Validations
        pred_spatial_shape = pred.shape[2:]
        target_spatial_shape = target.shape[-3:]
        if pred_spatial_shape != target_spatial_shape:
             raise ValueError(f"Spatial dimensions of pred {pred.shape} and target {target.shape} must match.")
        if self.monai_loss.to_onehot_y:
            target = target.unsqueeze(1) # ensure to [B, 1, Z, Y, X]
        
        if self.split_Z:
            z_slice_losses = [
                self.monai_loss(p, t) 
                for p, t in zip(pred.permute(2,0,1,3,4), target.permute(2,0,1,3,4))
            ]
            return torch.stack(z_slice_losses).mean()

        else:
            return self.monai_loss(pred, target)


class DiceLoss_3D(SplitZ_Loss):
    def __init__(self, loss_name = "loss_dice", **kwargs):
        from monai.losses.dice import DiceLoss
        super().__init__(split_Z=kwargs.pop("split_Z", False))
        self.loss_name = self._loss_name = loss_name
        self.monai_loss = DiceLoss(**kwargs)


class DiceCELoss_3D(SplitZ_Loss):
    def __init__(self, loss_name = "loss_DiceCE", **kwargs):
        from monai.losses.dice import DiceCELoss
        super().__init__(split_Z=kwargs.pop("split_Z", False))
        self.loss_name = self._loss_name = loss_name
        self.monai_loss = DiceCELoss(**kwargs)


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
