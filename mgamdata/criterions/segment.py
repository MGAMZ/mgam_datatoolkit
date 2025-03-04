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
        smooth_z_ratio:int|None = None,
        ignore_1st_index: bool = False,
        batch_z: int|None = None,
        eps=1e-5,
        ignore_index:int|None=None,
        loss_name="loss_dice",
    ):
        """
        Args:
            smooth_z_ratio: 沿着Z轴平滑Dice Loss的比例。默认: None
                            该值确定了有多大范围内的volume slice会被视为有效区域，
                            比例的基准是有效标注的Z轴长度，
                            在超出有效标注范围之后，损失权重逐渐降低至0，
                            降低至0的位置距离有效标注的中心Z切面是smooth_z_ratio * Z轴长度（注意，就是单侧）。
        """
        super().__init__()
        self.smooth_z_ratio = smooth_z_ratio
        self.eps = eps
        self.ignore_index = ignore_index
        self.loss_name = self._loss_name = loss_name
        self.ignore_1st_index = ignore_1st_index
        self.batch_z = batch_z

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
            one_hot_target.to(torch.int64), 
            num_classes + 1
        )
        one_hot_target = one_hot_target[..., :num_classes].permute(0, 4, 1, 2, 3)
        return one_hot_target

    def calc_dice_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        计算 Dice Loss，保持输入的维度结构。
        
        Args:
            pred (torch.Tensor): 预测张量，形状为 (n, c, h, w)
            target (torch.Tensor): 目标张量，形状为 (n, c, h, w)
            eps (float): 避免除零的小常数。默认: 1e-5
            ignore_index (int, optional): 需要忽略的类别索引。默认: None
            
        Returns:
            torch.Tensor: 未进行 reduction 的 Dice loss，与输入形状相同
        """
        if self.ignore_index is not None:
            num_classes = pred.shape[1]
            mask = torch.ones(num_classes, dtype=torch.bool, device=pred.device)
            mask[self.ignore_index] = False
            pred = pred[:, mask]
            target = target[:, mask]
            assert pred.shape[1] != 0, "忽略的索引后没有剩余类别"
        
        # 逐像素计算 intersection
        intersection = pred * target
        # 计算每个位置的平方和
        pred_square = pred * pred
        target_square = target * target
        # 计算 Dice coefficient
        numerator = 2 * intersection
        denominator = pred_square + target_square + self.eps
        return 1 - (numerator / denominator)

    # HACK for debug
    def visualize_z_loss(self, target, loss, z_weights):
        """
        可视化每个样本的loss沿Z方向上的sum大小，以及平滑权重
        
        Args:
            target: 形状为[N, C, Z, Y, X]或[N, Z, Y, X]的标注数据
            loss: 形状为[N, C, Z, Y, X]的损失值
            smooth_z_ratio: 平滑比例，如果为None则只显示原始loss
            z_weights: 预计算的Z轴权重，形状为[N, Z]，如果为None则根据smooth_z_ratio计算
            
        Returns:
            fig: matplotlib figure对象
        """
        from matplotlib import pyplot as plt
        assert self.smooth_z_ratio is not None
        
        if isinstance(target, torch.Tensor):
            target_np = target.detach().cpu().numpy()
        else:
            target_np = target
        if target_np.ndim == 5:
            target_np = target_np.any(axis=1)
            
        if isinstance(loss, Tensor):
            loss_np = loss.detach().cpu().numpy()
        else:
            loss_np = loss
        assert loss_np.ndim == 5
        
        # [N, Z]
        if isinstance(z_weights, Tensor):
            z_weights_np = z_weights.detach().cpu().numpy()
        else:
            z_weights_np = z_weights
        
        N, Z, Y, X = target_np.shape
        
        # 计算Z轴上的loss总和
        z_loss = loss_np.sum(axis=(1, 3, 4))  # [N, Z]
        
        # 找出每个样本中有标注的Z切片
        valid_z_mask = np.any(target_np, axis=(2, 3))  # [N, Z]
        
        # 创建子图
        fig, axes = plt.subplots(N, 1, figsize=(12, 4*N), squeeze=False)
        
        for n in range(N):
            ax = axes[n, 0]
            z_coords = np.arange(Z)
            
            # 绘制loss在Z轴上的分布
            ax.plot(z_coords, z_loss[n], 'b-', linewidth=2, label='Loss Sum')
            
            # 显示weights曲线
            ax2 = ax.twinx()
            ax2.plot(z_coords, z_weights_np[n], 'r--', linewidth=2, label='Weight')
            ax2.set_ylabel('Weight', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(0, 1.1)
            
            ax.set_xlabel('Z Position')
            ax.set_ylabel('Loss Sum', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            ax.set_title(f'Sample {n+1}')
            ax.set_xlim(0, Z)
            ax.grid(True)
            
            # 创建组合图例
            handles, labels = ax.get_legend_handles_labels()
            if z_weights_np is not None:
                handles2, labels2 = ax2.get_legend_handles_labels()
                handles += handles2
                labels += labels2
            ax.legend(handles, labels, loc='upper right')
        
        fig.tight_layout()
        fig.savefig(f'z_loss_visualization_{self.loss_name}.png')
        pdb.set_trace()
        exit()

    def get_smooth_z_weight(self, target:Tensor) -> Tensor:
        """
        根据target中有效标注的Z轴位置创建平滑权重，并应用于loss
        
        Args:
            target: 形状为[N, C, Z, Y, X]或[N, Z, Y, X]的标注数据
            
        Returns:
            z_weight: 形状为[N, Z]的权重张量
        """
        assert self.smooth_z_ratio is not None and self.smooth_z_ratio > 1
        N, Z = target.size(0), target.size(-3)
        device = target.device
        # 找出每个样本中有标注的Z切片
        valid_z_mask = target.any(dim=(1, 3, 4) if target.ndim==5 else (2,3))  # [N, Z]
        # 创建Z维度的权重张量
        z_weights = torch.zeros((N, Z), device=device)
        # Z轴坐标
        z_coords = torch.arange(Z, dtype=torch.float, device=device)
        
        for n in range(N):
            if valid_z_mask[n].any():  # 确保样本有有效标注
                # 找出有效Z的最小和最大索引
                z_indices = torch.where(valid_z_mask[n])[0]
                min_z = float(z_indices.min().item())
                max_z = float(z_indices.max().item())
                
                # 计算有效Z范围长度
                z_range = max_z - min_z + 1
                # 计算扩展范围
                extended_range = z_range * self.smooth_z_ratio
                half_extension = (extended_range - z_range) / 2
                # 计算扩展后的边界
                extended_min_z = min_z - half_extension
                extended_max_z = max_z + half_extension
                # 创建条件掩码
                left_region = (z_coords >= extended_min_z) & (z_coords < min_z)
                middle_region = (z_coords >= min_z) & (z_coords <= max_z)
                right_region = (z_coords > max_z) & (z_coords <= extended_max_z)
                # 自适应区域
                # 正梯形
                # z_weights[n, left_region] = (z_coords[left_region] - extended_min_z) / half_extension
                # z_weights[n, right_region] = 1 - (z_coords[right_region] - max_z) / half_extension
                # 楔形
                z_weights[n, left_region] = (1 - (z_coords[left_region] - extended_min_z) / half_extension) * 0.2 + 0.05
                z_weights[n, right_region] = (z_coords[right_region] - max_z) / half_extension * 0.2 + 0.05
                # 在有效标注区域内，权重为1
                z_weights[n, middle_region] = 1.0
        
        return z_weights
        
    def forward_one_patch(self, pred: Tensor, target: Tensor, *args, **kwargs):
        if pred.shape != target.shape:
            target = self._expand_onehot_labels_dice_3D(pred, target)
            assert pred.shape == target.shape
        # pred, target: [N, C, Z, Y, X]
        if self.ignore_1st_index:
            pred = pred[:, 1:, ...].contiguous()
            target = target[:, 1:, ...].contiguous()
        # [N, C, Z, Y, X]
        return self.calc_dice_loss(pred, target)

    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        # pred: [N, C, Z, Y, X]
        assert (pred.shape[-3:] == target.shape[-3:]), \
            f"The [Z, Y, X] of pred {pred.shape} and target {target.shape} must be the same."

        if self.batch_z is not None:
            batch_loss = []
            
            for z in range(0, pred.shape[-3], self.batch_z):
                batch_z_loss = self.forward_one_patch(
                    pred=pred[..., z : z + self.batch_z, :, :], 
                    target=target[..., z : z + self.batch_z, :, :], 
                    *args, **kwargs
                )
                batch_loss.append(batch_z_loss)

            # [N, C, Z, Y, X]
            loss = torch.concatenate(batch_loss, dim=-3)

        else:
            # [N, C, Z, Y, X]
            loss = self.forward_one_patch(pred, target, *args, **kwargs)

        if self.smooth_z_ratio is not None:
            z_weights = self.get_smooth_z_weight(target)
            # self.visualize_z_loss(target, loss, z_weights) # HACK debug
            weights = z_weights.view(z_weights.size(0), 1, z_weights.size(2), 1, 1).expand_as(loss)
            weighted_loss = (loss * weights)
        
        return weighted_loss.mean()


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
