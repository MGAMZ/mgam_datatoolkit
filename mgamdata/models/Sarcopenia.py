import pdb
import warnings
from collections.abc import Sequence
from typing_extensions import deprecated

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from matplotlib.patches import Patch
from scipy.ndimage import zoom

from mmcv.transforms import BaseTransform
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from mmengine.evaluator.metric import BaseMetric
from mmengine.runner import Runner
from ..mm.mmseg_PlugIn import IoUMetric_PerClass
from ..mm.mmseg_Dev3D import (BaseDecodeHead_3D, Seg3DDataSample, Seg3DDataPreProcessor,
                              PixelShuffle3D, EncoderDecoder_3D, VolumeData)
from ..mm.visualization import BaseViser, BaseVisHook
from ..mm.inference import Inferencer
from ..mm.mgam_models import mgam_Seg3D_Lite



class SeriesData(BaseDataElement):
    """Data structure for 1D annotations or predictions.

    所有存储在 ``data_fields`` 中的数据都满足以下要求：
    - 维度为 [C(可选), Z]。
    - 它们应当具有相同的 Z 维度。
    """

    def __setattr__(self, name: str, value: Tensor | np.ndarray):
        """Set attributes of ``VolumeData``.

        若传入数据为 1D（即形状仅 [Z]），则自动加一维成为 [1, Z]。

        Args:
            name (str): 设置属性的名称。
            value (Union[Tensor, np.ndarray]): 要存储的值，只能是
                `Tensor` 或 `np.ndarray`，形状必须满足 [C(可选), Z] 的要求。
        """
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f"{name} 已被用作私有属性，无法再次修改。")
        else:
            assert isinstance(value, (Tensor, np.ndarray)), (
                f"无法设置 {type(value)}，仅支持 {(Tensor, np.ndarray)}。")

            if self.shape:
                # 只比较最后一维是否一致
                assert value.shape[-1] == self.shape[0], (
                    f"传入数据的 Z 维度 {value.shape[-1]} "
                    f"与当前 VolumeData 的 shape {self.shape} 不一致。"
                )

            # 检查维度必须为 1 或 2
            assert value.ndim in [1, 2], f"数据维度必须为 1 或 2，但当前为 {value.ndim}。"
            if value.ndim == 1:
                value = value[None]  # 在最前面加一维
                warnings.warn(f"已将输入从 (Z,) 自动转换为 (1, Z)。当前形状为 {value.shape}。")

            super().__setattr__(name, value)

    def __getitem__(self, item: int | slice) -> "VolumeData":
        """
        Args:
            item (Union[int, slice]): 根据下标或切片从 Z 维度获取数据。

        Returns:
            :obj:`VolumeData`: 切片后的数据。
        """
        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, (int, slice)):
            for k, v in self.items():
                setattr(new_data, k, v[:, item])
        else:
            raise TypeError(f"仅支持 int 或 slice 类型的索引，但传入 {type(item)}。")
        return new_data

    @property
    def shape(self):
        """返回当前数据的 Z 维度，比如 (100,)。忽略通道维度。"""
        if len(self._data_fields) > 0:
            # 只返回最后一维的大小
            return (self.values()[0].shape[-1],)
        else:
            return None


class SarcopeniaPreprocessor(Seg3DDataPreProcessor):
    @staticmethod
    def stack_batch_3D(
        inputs: list[Tensor],
        data_samples: list[Seg3DDataSample] | None = None,
        size: tuple | None = None,
        size_divisor: int | None = None,
        pad_val: int | float = 0,
        seg_pad_val: int | float = 255,
    ):
        assert isinstance(
            inputs, list
        ), f"Expected input type to be list, but got {type(inputs)}"
        assert len({tensor.ndim for tensor in inputs}) == 1, (
            f"Expected the dimensions of all inputs must be the same, "
            f"but got {[tensor.ndim for tensor in inputs]}"
        )
        assert inputs[0].ndim == 4, (
            f"Expected tensor dimension to be 4, " f"but got {inputs[0].ndim}"
        )
        assert len({tensor.shape[0] for tensor in inputs}) == 1, (
            f"Expected the channels of all inputs must be the same, "
            f"but got {[tensor.shape[0] for tensor in inputs]}"
        )

        # only one of size and size_divisor should be valid
        assert (size is not None) ^ (
            size_divisor is not None
        ), "only one of size and size_divisor should be valid"

        padded_inputs = []
        padded_samples = []
        inputs_sizes = [(img.shape[-3], img.shape[-2], img.shape[-1]) for img in inputs]
        max_size = np.stack(inputs_sizes).max(0)
        if size_divisor is not None and size_divisor > 1:
            # the last three dims are Z,Y,X, all subject to divisibility requirement
            max_size = (max_size + (size_divisor - 1)) // size_divisor * size_divisor

        for i in range(len(inputs)):
            tensor = inputs[i]
            if size is not None:
                if len(size) == 2:
                    size = (tensor.shape[-3], *size)
                depth = max(size[-3] - tensor.shape[-3], 0)
                height = max(size[-2] - tensor.shape[-2], 0)
                width = max(size[-1] - tensor.shape[-1], 0)
                # (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
                padding_size = (0, width, 0, height, 0, depth)
            elif size_divisor is not None:
                depth = max(max_size[-3] - tensor.shape[-3], 0)
                height = max(max_size[-2] - tensor.shape[-2], 0)
                width = max(max_size[-1] - tensor.shape[-1], 0)
                padding_size = (0, width, 0, height, 0, depth)
            else:
                padding_size = (0, 0, 0, 0, 0, 0)

            # pad volume
            pad_volume = torch.nn.functional.pad(tensor, padding_size, value=pad_val)
            padded_inputs.append(pad_volume)
            # pad gt_sem_seg
            if data_samples is not None:
                data_sample = data_samples[i]
                pad_shape = None
                if "gt_sem_seg" in data_sample:
                    gt_sem_seg = data_sample.gt_sem_seg.data
                    del data_sample.gt_sem_seg.data
                    data_sample.gt_sem_seg.data = torch.nn.functional.pad(gt_sem_seg, padding_size, value=seg_pad_val)
                    pad_shape = data_sample.gt_sem_seg.shape
                if "gt_sem_seg_one_hot" in data_sample:
                    gt_sem_seg_one_hot = data_sample.gt_sem_seg_one_hot.data
                    del data_sample.gt_sem_seg_one_hot.data
                    data_sample.gt_sem_seg_one_hot.data = torch.nn.functional.pad(gt_sem_seg_one_hot, padding_size, value=0)
                
                # NOTE This is the ONLY difference between sarcopenia and other preprocessor.
                if "gt_L3" in data_sample:
                    gt_L3 = data_sample.gt_L3.data
                    del data_sample.gt_L3.data
                    
                    data_sample.gt_L3.data = torch.nn.functional.pad(gt_L3, padding_size[-2:], value=0)
                
                data_sample.set_metainfo(
                    {
                        "img_shape": tensor.shape[-3:],
                        "pad_shape": pad_shape,
                        "padding_size": padding_size,
                    }
                )
                padded_samples.append(data_sample)
            
            else:
                padded_samples.append({
                    "img_padding_size": padding_size,
                    "pad_shape": pad_volume.shape[-3:]
                })

        return torch.stack(padded_inputs, dim=0), padded_samples


class L3_Evaluator(BaseMetric):
    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        acc = []
        for sample in data_samples:
            pred = sample.get("pred_L3")
            gt = sample.get("gt_L3")
            acc.append((pred == gt).float())
        self.results.extend(acc)
    
    def compute_metrics(self, results: list) -> dict:
        return {"Val/acc_L3": np.mean(results)}


class L3_VisHook(BaseVisHook):
    def __init__(self, interval:int=1, draw:bool=True):
        self.interval = interval
        self.draw = draw
    
    def after_val_iter(self,
                       runner:Runner,
                       batch_idx: int,
                       data_batch: list[BaseDataElement],
                       outputs: list[BaseDataElement]) -> None:
        """
        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        if self.draw is False or batch_idx % self.interval != 0:
            return
        
        # 创建具有多个子图的图像，每个子图对应一个输出
        if len(outputs) == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 2))
            axes = [axes]
        else:
            fig, axes = plt.subplots(len(outputs), 1, figsize=(10, len(outputs)*2))
        
        # 逐样本绘制图像
        for i, (output, ax) in enumerate(zip(outputs, axes)):
            pred = output.get_field("pred_L3")  # [Z]
            gt = output.get_field("gt_L3")  # [Z]
            
            # 确保转换为numpy数组以便visualization
            if not isinstance(pred, np.ndarray):
                pred = np.array(pred)
            if not isinstance(gt, np.ndarray):
                gt = np.array(gt)
                
            # 创建一个2行的数组，第一行是pred，第二行是gt
            display_data = np.vstack((pred, gt))
            
            # 使用imshow绘制数据，binary cmap会将0显示为白色，1显示为黑色
            im = ax.imshow(display_data, aspect='auto', cmap='binary', interpolation='none')
            
            # 添加y轴标签
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Pred', 'GT'])
            
            # 添加标题
            ax.set_title(f'Sample {i+1}')
            
            # 如果序列很长，可以简化x轴刻度
            if len(pred) > 20:
                ax.set_xticks(np.arange(0, len(pred), len(pred)//10))
        
        fig.tight_layout()
        fig.canvas.draw()  # 先绘制图形到画布
        width, height = fig.get_size_inches() * fig.get_dpi()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(int(height), int(width), 3)
        plt.close(fig)
        runner.visualizer.add_image("Pred_L3", img_array, global_step=runner.iter)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: list[BaseDataElement],
                        outputs: list[BaseDataElement]) -> None:
        """
        Args:
            runner (Runner): The runner of the training  process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """


class L3_Visualizer(BaseViser):
    def __init__(
        self,
        name,
        resize: Sequence[int] | None = None,
        label_text_scale: float = 0.05,
        label_text_thick: float = 1,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        self.resize = resize
        self.label_text_scale = label_text_scale
        self.label_text_thick = label_text_thick

    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Seg3DDataSample | None = None,
        *args,
        **kwargs,
    ) -> None:
        """显示ZY横断面，取X的中心切面，并可视化1D的L3标记数据。

        Args:
            name: 图像的名称
            image (np.ndarray): 要可视化的图像，NdArray (Z, Y, X, C)。
            data_sample (Seg3DDataSample, optional): 要可视化的数据样本。
                - gt_L3 (data:SeriesData): tensor (Z,)
                - pred_sem_seg_L3 (data:SeriesData): tensor (Z,)
        """
        assert image.ndim == 4, (
            f"The input image must be 4D, but got " f"shape {image.shape}."
        )
        Z, Y, X, C = image.shape
        name += f"_zy_plane"
        
        # 取X的中心切面，显示ZY横断面
        center_x = X // 2
        zy_plane = image[:, :, center_x, :].copy()
        zy_plane = (zy_plane / zy_plane.max() * 255).astype(np.uint8)  # (Z, Y, C)
        
        if self.resize is not None:
            zy_plane = cv2.resize(zy_plane, self.resize, interpolation=cv2.INTER_LINEAR)
        
        l3_info = {'gt_L3': data_sample.gt_L3.data.squeeze(0),
                   'pred_L3': data_sample.pred_sem_seg_L3.data.squeeze(0)}
        
        self.draw_L3(name, zy_plane, l3_info, *args, **kwargs)
    
    def draw_L3(self, name, image, l3_info, *args, **kwargs):
        """绘制1D的L3标记在ZY平面上的可视化结果。
        
        Args:
            name: 图像名称
            image: ZY平面的图像 (Z, Y, C)
            l3_info: 包含gt_L3和pred_L3的字典
        """
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(image) # 显示ZY平面图像
        
        if 'gt_L3' in l3_info:
            orig_Z = len(l3_info['gt_L3'])
        elif 'pred_L3' in l3_info:
            orig_Z = len(l3_info['pred_L3'])
        else:
            ax.set_title('没有L3数据可用')
            fig.canvas.draw()
            img_array = self.export_fig_to_ndarray(fig)
            self.add_image(name, img_array, step=kwargs.get('step', 0))
            return
        
        # 计算缩放比例，缩放在外层调用进行
        Z, Y, _ = image.shape
        scale_factor = Z / orig_Z
        
        # 创建一个叠加图层用于GT和Pred
        gt_overlay = np.zeros((Z, Y, 4), dtype=np.uint8)  # RGBA
        pred_overlay = np.zeros((Z, Y, 4), dtype=np.uint8)  # RGBA
        
        # 绘制gt_L3
        gt_l3 = l3_info['gt_L3']
        if isinstance(gt_l3, torch.Tensor):
            gt_l3 = gt_l3.cpu().numpy()
        gt_l3_resize_to_img = zoom(gt_l3, zoom=scale_factor, order=0)
        gt_positions = np.where(gt_l3_resize_to_img == 1)[0]
        if len(gt_positions) > 0:
            for scaled_pos in gt_positions:
                # 半透明绿色
                gt_overlay[scaled_pos, :, 0] = 0    # R
                gt_overlay[scaled_pos, :, 1] = 255  # G
                gt_overlay[scaled_pos, :, 2] = 0    # B
                gt_overlay[scaled_pos, :, 3] = 255  # A
        ax.imshow(gt_overlay, alpha=0.5)

        # 绘制pred_L3
        pred_l3 = l3_info['pred_L3']
        if isinstance(pred_l3, torch.Tensor):
            pred_l3 = pred_l3.cpu().numpy()
        pred_l3_resize_to_img = zoom(pred_l3, zoom=scale_factor, order=0)
        pred_positions = np.where(pred_l3_resize_to_img == 1)[0]
        try:
            if len(pred_positions) > 0:
                for scaled_pos in pred_positions:
                    # 半透明红色
                    pred_overlay[scaled_pos, :, 0] = 255  # R
                    pred_overlay[scaled_pos, :, 1] = 0    # G
                    pred_overlay[scaled_pos, :, 2] = 0    # B
                    pred_overlay[scaled_pos, :, 3] = 255  # 不透明
            ax.imshow(pred_overlay, alpha=0.5)
        except Exception as e:
            pdb.set_trace()
            pass
        
        # 图像元素
        legend_elements = [Patch(facecolor='green', alpha=0.3, label='GT L3')]
        legend_elements.append(Patch(facecolor='red', label='Pred L3'))
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        fig.tight_layout()
        
        # 输出
        fig.canvas.draw()
        img_array = self.export_fig_to_ndarray(fig)
        self.add_image(name, img_array, step=kwargs.get('step', 0))


class L3Metric(BaseMetric):
    """
    1D场景下的Recall、Precision和HD95计算。
    使用pred_sem_seg_L3和gt_sem_seg_L3分别表示预测和真值。
    """
    def process(self, data_batch: dict, data_samples: list[dict]) -> None:
        """
        收集每批数据中的预测和真值。
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg_L3']['data'].squeeze()
            gt_label = data_sample['gt_L3']['data'].squeeze()
            pred_label = pred_label if isinstance(pred_label, np.ndarray) else pred_label.cpu().numpy()
            gt_label = gt_label if isinstance(gt_label, np.ndarray) else gt_label.cpu().numpy()
            assert pred_label.shape == gt_label.shape, (
                f"预测和真值的形状不一致：{pred_label.shape} != {gt_label.shape}")
            
            self.results.append((pred_label, gt_label))

    def compute_metrics(self, results) -> dict[str, float]:
        """
        计算1D场景下的Recall、Precision和HD95。
        """
        ious, recalls, precisions, hd95s, productCompliance = [], [], [], [], []

        for (pred, gt) in results:
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))
            
            iou = tp / max(tp + fp + fn, 1e-6)
            precision = tp / max(tp + fp, 1e-6)
            recall = tp / max(tp + fn, 1e-6)
            hd95 = self.compute_hd95_1d(pred, gt)

            ious.append(iou)
            recalls.append(recall)
            precisions.append(precision)
            hd95s.append(hd95)
            productCompliance.append(iou>=0.85)

        return {
            'iou': float(np.mean(ious)),
            'Recall': float(np.mean(recalls)),
            'Precision': float(np.mean(precisions)),
            'HD95': float(np.mean(hd95s)),
            'ProductComplianceRatio': float(np.mean(productCompliance))
        }

    def compute_hd95_1d(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        在1D数组中计算Hausdorff Distance的95分位数(HD95)。

        1. 找到pred和gt为1的所有索引，构成前景位置。
        2. 对双方前景位置计算最近距离并合并。
        3. 返回这些距离的95分位数。
        """
        pred_indices = np.where(pred == 1)[0]
        gt_indices = np.where(gt == 1)[0]

        # 若任一前景为空，约定返回0
        if len(pred_indices) == 0 and len(gt_indices) == 0:
            return 0.0
        if len(pred_indices) == 0 or len(gt_indices) == 0:
            return float(max(len(pred), len(gt)))

        # 分别计算 pred->gt, gt->pred 的最近距离
        dist_list = []
        for p in pred_indices:
            dist_list.append(np.min(np.abs(gt_indices - p)))
        for g in gt_indices:
            dist_list.append(np.min(np.abs(pred_indices - g)))

        # 返回95分位数距离
        return float(np.percentile(dist_list, 95))


class gen_L3_label(BaseTransform):
    """
    Required Fields:
        - img: [Z, Y, X, C (May Exist)]
    
    Added Fields:
        - gt_L3: [1, Z]
    """
    def transform(self, results:dict):
        if 'L3_anno' in results:
            Z, Y, X = results['img'].shape[:3]
            mask = np.zeros((1, Z), dtype=np.uint8)
            anno_L3_start, anno_L3_mid, anno_L3_end = results['L3_anno']
            # 鉴影标号是倒序的 NOTE 注意下方start和end调转匹配了，不是简单的减Z
            # anno_L3_end, anno_L3_mid, anno_L3_start = Z - anno_L3_start, Z - anno_L3_mid, Z - anno_L3_end
            mask[0, anno_L3_start:anno_L3_end] = 1
            results['gt_L3'] = mask
            results['seg_fields'].append('gt_L3')
        
        elif 'gt_seg_map' in results:
            results['gt_L3'] = np.any(results['gt_seg_map'], axis=(1,2))[None].astype(np.uint8) # pyright:ignore
            results['seg_fields'].append('gt_L3')
        
        return results


class ForegroundSlicesMetric(IoUMetric_PerClass):
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes']) # pyright:ignore
        for data_sample in data_samples:
            pred_label:Tensor = data_sample['pred_sem_seg']['data'].squeeze()         # [Z, Y, X]
            label:Tensor = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label) # [Z, Y, X]
            foreground_Zs = label.any(dim=(1,2)).argwhere()
            pred_label = pred_label[foreground_Zs]
            label = label[foreground_Zs]
            batch_result = self.intersect_and_union(pred_label, label, num_classes, self.ignore_index)
            self.results.append(batch_result)



""" ----- Neural Models ----- """


class L3LocationDecoder(BaseDecodeHead_3D):
    def __init__(self, 
                 embed_dims:list[int], 
                 Z_lengths:list[int], 
                 threshold:float=0.3, 
                 loss_weight:float=1., 
                 use_checkpoint:bool=False,
                 pixel_shuffle:int|None=None,
                 detach_with_encoder:bool=True,
                 *args, **kwargs):
        assert len(embed_dims) == len(Z_lengths)
        super().__init__(in_channels=embed_dims,
                         channels=1,
                         num_classes=1,
                         in_index=list(range(len(embed_dims))),
                         input_transform='multiple_select',
                         threshold=threshold,
                         *args, **kwargs
        )
        self.embed_dims = embed_dims
        self.Z_lengths = Z_lengths
        self.threshold = threshold
        self.loss_weight = loss_weight
        self.use_checkpoint = use_checkpoint
        self.detach_with_encoder = detach_with_encoder
        
        num_layers = 4
        assert Z_lengths[0] == Z_lengths[-1]*(2**num_layers), \
            f"Source Z length ({Z_lengths[0]}) should be equal to that of extracted feats ({Z_lengths[-1]*(2**num_layers)})."

        if pixel_shuffle is not None and pixel_shuffle > 1:
            self.pixel_shuffle = PixelShuffle3D(pixel_shuffle)

        # extraction and reduce to Z
        self.downsample = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in_C = embed_dims[-1] // (2**layer_idx)
            layer_out_C = embed_dims[-1] // (2**(layer_idx+1))
            self.downsample.extend([
                torch.nn.Conv3d(in_channels=layer_in_C, out_channels=layer_in_C, kernel_size=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv3d(in_channels=layer_in_C, out_channels=layer_in_C, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose3d(in_channels=layer_in_C, out_channels=layer_out_C, kernel_size=(2,1,1), stride=(2,1,1)),
                torch.nn.LeakyReLU(),
                torch.nn.GroupNorm(num_groups=8, num_channels=layer_out_C)
            ])
        
        # decide
        self.XY_pooling = torch.nn.AdaptiveMaxPool3d(output_size=(Z_lengths[0], 1, 1))
        self.foreground_decider = torch.nn.Conv1d(in_channels=layer_out_C, out_channels=1, kernel_size=1)
    
    @staticmethod
    def is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0

    def forward(self, x: list[Tensor]) -> Tensor:
        assert x[0].ndim == 5, "Input tensor should be 5D tensor, [B, C, Z, H, W], BUT got {}".format(x[0].shape)
        assert len(x) == len(self.embed_dims), "Input tensor should have {} layers, BUT got {}".format(len(self.embed_dims), len(x))
        
        feat = x[-1] # [B, C, Z, H, W]
        if self.detach_with_encoder:
            # NOTE Detach from the encoder, not influencing the encoder, improve stability.
            feat.detach_() # [B, C, Z, H, W]
        
        if hasattr(self, "pixel_shuffle"):
            feat = self.pixel_shuffle(feat)
        
        # transfer feat from X Y to Z
        for layer in self.downsample:
            feat = layer(feat)
        feat = self.XY_pooling(feat) # [B, C, Z, 1, 1]
        feat = feat.squeeze(-1).squeeze(-1) # [B, C, Z]
        
        # decider, [B, 1, Z]
        if self.use_checkpoint:
            feat = torch.utils.checkpoint.checkpoint(self.foreground_decider, feat)
        else:
            feat = self.foreground_decider(feat)

        return feat.squeeze(1) # [B, Z]

    def loss(
        self,
        inputs: list[Tensor],
        batch_data_samples: list[Seg3DDataSample],
        train_cfg:dict|None=None,
    ) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]):
                list of multi-level img features.
                (N, C, Z, Y, X)

            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.

            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        foreground_Zs = self._stack_batch_gt(batch_data_samples, "gt_L3").float() # [B, Z]
        
        z_results = self.forward(inputs) # [B, Z]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z_results, foreground_Zs)
        
        with torch.no_grad():
            pred = z_results > self.threshold
            foreground_Zs = foreground_Zs.bool()
            hit = (pred == foreground_Zs).float().mean()
            iou = (pred & foreground_Zs).sum() / ((pred | foreground_Zs).sum() + 1)
        
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(hit, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(iou, op=torch.distributed.ReduceOp.AVG)
        
        return {"loss_L3": loss * self.loss_weight,
                "acc_L3": hit,
                "iou_L3": iou}
    
    def predict(self, 
                inputs:list[Tensor], 
                data_samples:list[BaseDataElement]|None=None, 
                test_cfg:dict|None=None) -> Tensor:
        z_logits = self.forward(inputs)
        return z_logits > self.threshold


class L3Locator(BaseModel):
    def __init__(self, backbone:dict, locate_head:dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = MODELS.build(backbone)
        self.locate_head = MODELS.build(locate_head)

    def gen_Z_label(self, data_samples):
        for i, sample in enumerate(data_samples):
            ann = sample.gt_sem_seg.data # [Z, Y, X]
            if ann.ndim == 4:
                ann = ann.squeeze(0) # [C, Z, Y, X] -> [Z, Y, X]
            z_ann = torch.any(ann, dim=(1,2)).float() # [Z]
            data_samples[i].set_field(z_ann, "gt_L3")
        return data_samples

    def loss(self, inputs, data_samples) -> dict:
        feat = self.backbone(inputs)
        loss = self.locate_head.loss(feat, data_samples)
        return loss
    
    def encode_decode(self, inputs, data_samples) -> Tensor:
        feat = self.backbone(inputs)
        return self.locate_head.predict(feat, data_samples)
    
    def slide_inference(
        self,
        inputs: Tensor,
        batch_img_metas: list[BaseDataElement],
    ) -> Tensor:
        """Inference by sliding-window with overlap.

        If z_crop > z_img or y_crop > y_img or x_crop > x_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxZxYxX,
                which contains all volumes in the batch.
            batch_img_metas (list[dict]): list of volume metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input volume.
        """
        assert self.test_cfg.mode == "slide", "Only support slide mode, got {}".format(self.test_cfg.mode)

        accu_device: str = self.test_cfg.slide_accumulate_device
        z_stride, y_stride, x_stride = self.test_cfg.stride  # type: ignore
        z_crop, y_crop, x_crop = self.test_cfg.crop_size  # type: ignore
        batch_size, _, z_img, y_img, x_img = inputs.size()
        out_channels = self.out_channels
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(y_img - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(x_img - x_crop + x_stride - 1, 0) // x_stride + 1
        preds = torch.zeros(
            size=(batch_size, out_channels, z_img, y_img, x_img),
            dtype=torch.float16,
            device=accu_device,
            pin_memory=False,
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, z_img, y_img, x_img),
            dtype=torch.uint8,
            device=accu_device,
            pin_memory=False,
        )

        for z_idx in range(z_grids):
            for y_idx in range(y_grids):
                for x_idx in range(x_grids):
                    z1 = z_idx * z_stride
                    y1 = y_idx * y_stride
                    x1 = x_idx * x_stride
                    z2 = min(z1 + z_crop, z_img)
                    y2 = min(y1 + y_crop, y_img)
                    x2 = min(x1 + x_crop, x_img)
                    z1 = max(z2 - z_crop, 0)
                    y1 = max(y2 - y_crop, 0)
                    x1 = max(x2 - x_crop, 0)
                    crop_vol = inputs[:, :, z1:z2, y1:y2, x1:x2]
                    # change the volume shape to patch shape
                    batch_img_metas[0]["img_shape"] = crop_vol.shape[2:] # type: ignore
                    # the output of encode_decode is seg logits tensor map
                    # with shape [N, C, Z, Y, X]
                    # NOTE WARNING:
                    # Setting `non_blocking=True` WILL CAUSE:
                    # Invalid pred_seg_logit accumulation on X axis.
                    crop_seg_logit = self.encode_decode(crop_vol, batch_img_metas).to(
                        accu_device, non_blocking=False)
                    preds[:, :, z1:z2, y1:y2, x1:x2] += crop_seg_logit
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1

        assert torch.all(count_mat != 0), "The count_mat should not be zero"
        seg_logits = preds / count_mat
        return seg_logits

    def forward(self,
                inputs: torch.Tensor,
                data_samples: list[BaseDataElement],
                mode: str = 'tensor'):
        """
            - If ``mode == loss``, return a ``dict`` of loss tensor used for backward and logging.
            - If ``mode == predict``, return a ``list`` of inference results.
            - If ``mode == tensor``, return a tensor or ``tuple`` of tensor or ``dict`` of tensor for custom use.
        """
        
        data_samples = self.gen_Z_label(data_samples)

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            z_pred = self.slide_inference(inputs, data_samples)
            for prediction, data_sample in zip(z_pred, data_samples):
                data_sample.set_field(prediction, "pred_L3")
            return data_samples
        elif mode == 'tensor':
            return self.encode_decode(inputs, data_samples)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")


class SarcopeniaSegmentorWithL3Locating(EncoderDecoder_3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.with_auxiliary_head, "The model should have auxiliary head, it will locate L3."

    def L3_filter(self, encoder_out, batch_img_metas):
        self.auxiliary_head: L3LocationDecoder
        L3_location = self.auxiliary_head.predict(encoder_out, batch_img_metas) # [B, Z]
        B, Z = L3_location.shape
        continuous_L3_location = torch.zeros_like(L3_location)

        for b in range(B):
            # 找到掩码中值为 1 的最小和最大下标
            indices = torch.nonzero(L3_location[b], as_tuple=False).squeeze()
            if indices.numel() == 0:
                # 如果该 batch 没有值为 1 的元素，跳过
                continue
            min_idx = indices.min()
            max_idx = indices.max()

            # 将最小和最大下标之间的区间置为 1
            continuous_L3_location[b, min_idx:max_idx+1] = 1

        return continuous_L3_location[:, None, :, None, None] # [B, 1, Z, 1, 1]

    def slide_inference(
        self,
        inputs: Tensor,
        batch_img_metas: list[dict],
    ) -> Tensor:
        """Inference by sliding-window with overlap.

        If z_crop > z_img or y_crop > y_img or x_crop > x_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxZxYxX,
                which contains all volumes in the batch.
            batch_img_metas (list[dict]): list of volume metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input volume.
        """

        accu_device: str = self.test_cfg.slide_accumulate_device
        z_stride, y_stride, x_stride = self.test_cfg.stride  # type: ignore
        z_crop, y_crop, x_crop = self.test_cfg.crop_size  # type: ignore
        batch_size, _, z_img, y_img, x_img = inputs.size()
        out_channels = self.out_channels
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(y_img - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(x_img - x_crop + x_stride - 1, 0) // x_stride + 1
        preds = torch.zeros(
            size=(batch_size, out_channels, z_img, y_img, x_img),
            dtype=torch.float16,
            device=accu_device,
            pin_memory=False,
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, z_img, y_img, x_img),
            dtype=torch.uint8,
            device=accu_device,
            pin_memory=False,
        )

        for z_idx in range(z_grids):
            for y_idx in range(y_grids):
                for x_idx in range(x_grids):
                    z1 = z_idx * z_stride
                    y1 = y_idx * y_stride
                    x1 = x_idx * x_stride
                    z2 = min(z1 + z_crop, z_img)
                    y2 = min(y1 + y_crop, y_img)
                    x2 = min(x1 + x_crop, x_img)
                    z1 = max(z2 - z_crop, 0)
                    y1 = max(y2 - y_crop, 0)
                    x1 = max(x2 - x_crop, 0)
                    crop_vol = inputs[:, :, z1:z2, y1:y2, x1:x2]
                    # change the volume shape to patch shape
                    batch_img_metas[0]["img_shape"] = crop_vol.shape[2:]
                    # the output of encode_decode is seg logits tensor map
                    # with shape [N, C, Z, Y, X]
                    # NOTE WARNING:
                    # Setting `non_blocking=True` WILL CAUSE:
                    # Invalid pred_seg_logit accumulation on X axis.
                    encoder_out = self.extract_feat(crop_vol)
                    seg_logits = self.decode_head.predict(encoder_out, batch_img_metas, self.test_cfg) # [B, C, Z, Y, X]
                    L3_mask = self.L3_filter(encoder_out, batch_img_metas) # [B, 1, Z, 1, 1]
                    seg_logits_foreground = (seg_logits*L3_mask).to(accu_device, non_blocking=False)
                    preds[:, :, z1:z2, y1:y2, x1:x2] += seg_logits_foreground
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1

        assert torch.all(count_mat != 0), "The count_mat should not be zero"
        seg_logits = preds / count_mat
        return seg_logits


class SarcopeniaL3Locating(EncoderDecoder_3D):
    """Encoder Decoder segmentors for 3D data, modified to output [B, Z] binary masks."""

    def slide_inference(
        self,
        inputs: Tensor,
        batch_img_metas: list[dict],
    ) -> Tensor:
        """Inference by sliding-window only on the Z axis with overlap.

        1. 仅在 Z 轴维度进行分块，其余维度 (X, Y) 均使用完整切片；
        2. 假定模型的输出形状与分块后的输入相匹配，且最终只需在 Z 维度进行拼接。
        """
        accu_device: str = self.test_cfg.slide_accumulate_device
        z_stride = self.test_cfg.stride[0]  # 仅使用 Z 方向的 stride # type: ignore
        z_crop = self.test_cfg.crop_size[0]  # 仅使用 Z 方向的 crop # type: ignore
        batch_size, _, z_img, y_img, x_img = inputs.size()

        # 计算 Z 轴上需要多少次分块
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1

        # 用于累计分块结果的容器，最终形状为 [B, Z]
        preds = torch.zeros(
            size=(batch_size, z_img),
            dtype=torch.float16,
            device=accu_device,
            pin_memory=False,
        )
        # 记录每个 Z 位置被累加的次数
        count_mat = torch.zeros(
            size=(batch_size, z_img),
            dtype=torch.uint8,
            device=accu_device,
            pin_memory=False,
        )

        # 仅在 Z 轴方向进行分块
        for z_idx in range(z_grids):
            z1 = z_idx * z_stride
            z2 = min(z1 + z_crop, z_img)
            z1 = max(z2 - z_crop, 0)

            # 提取整块 [Z1:Z2, :, :], 不在 Y, X 方向做切分
            crop_vol = inputs[:, :, z1:z2, :, :]

            # 设置对应分块形状 (仅修改 Z 方向)
            batch_img_metas[0]["img_shape"] = crop_vol.shape[2:3]
            batch_img_metas[0]["pad_shape"] = crop_vol.shape[2:3]

            # 推理得到形状 [B, 1, z_patch, 1, 1]
            crop_seg_logit = self.encode_decode(crop_vol, batch_img_metas)
            crop_seg_logit = crop_seg_logit.to(accu_device, non_blocking=False)

            # 将分块结果累加到 preds 中，将输出 squeeze 成 [B, z_patch]
            crop_seg_logit = crop_seg_logit.squeeze(1).squeeze(-1).squeeze(-1)  # [B, z_patch]
            preds[:, z1:z2] += crop_seg_logit
            count_mat[:, z1:z2] += 1

        assert torch.all(count_mat != 0), "The count_mat should not contain zero"
        seg_logits = preds / count_mat  # [B, Z]
        return seg_logits

    def postprocess_result(
        self, 
        seg_logits: Tensor, 
        data_samples: list[Seg3DDataSample]
    ) -> list[Seg3DDataSample]:
        
        batch_size, z_len = seg_logits.shape

        for i in range(batch_size):
            # 对 seg_logits 进行阈值处理，得到 [B, Z] 的二值结果
            i_seg_logits = seg_logits[i]  # [Z]
            i_seg_logits = i_seg_logits.sigmoid()
            i_seg_pred = (i_seg_logits > self.decode_head.threshold).to(torch.uint8)
            data_samples[i].set_data({
                "seg_logits_L3": SeriesData(**{"data": i_seg_logits.unsqueeze(0)}),  # [1, Z]
                "pred_sem_seg_L3": SeriesData(**{"data": i_seg_pred.unsqueeze(0)}),  # [1, Z]
            })

        return data_samples


class L3LocatingLite(mgam_Seg3D_Lite):
    def _forward(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        return self.backbone(inputs).mean(dim=(1,3,4)) # [B, C, Z, Y, X] -> [B, Z]

    def predict(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Sequence[BaseDataElement]:
        """Predict results from a batch of inputs and data samples.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, Z, Y, X).
            data_samples (Sequence[BaseDataElement], optional): The seg data samples.
                It usually includes information such as `metainfo`.
                
        Returns:
            Sequence[BaseDataElement]: Segmentation results of the input images.
                Each SegDataSample usually contains:
                - pred_sem_seg (VolumeData): Prediction of semantic segmentation.
                - seg_logits (VolumeData): Predicted logits of semantic segmentation.
        """
        # 前向传播
        seg_logits = self.inference(inputs, data_samples) # [N, C, Z, Y, X]
        
        # 处理结果
        batch_size = inputs.shape[0]
        out_channels = seg_logits.shape[1]
        
        # 验证二分类阈值与模型输出通道数的一致性
        if out_channels > 1 and self.binary_segment_threshold is not None:
            raise ValueError(f"多分类模型(输出通道数={out_channels})不应设置binary_segment_threshold，"
                            f"当前值为{self.binary_segment_threshold}，应设置为None")
        if out_channels == 1 and self.binary_segment_threshold is None:
            raise ValueError(f"二分类模型(输出通道数={out_channels})必须设置binary_segment_threshold，"
                            "当前值为None")
        
        if data_samples is None:
            data_samples = [BaseDataElement() for _ in range(batch_size)]
        
        for i in range(batch_size):
            # 处理单个样本
            i_seg_logits = seg_logits[i] # [C, Z, Y, X]
            
            # 生成预测结果
            if out_channels > 1:  # 多分类情况
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:  # 二分类情况
                assert self.binary_segment_threshold is not None, \
                    f"二分类模型(输出通道数={out_channels})必须设置binary_segment_threshold，" \
                    f"当前值为None"
                i_seg_logits_sigmoid = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits_sigmoid > self.binary_segment_threshold).to(i_seg_logits)
            
            # 将结果保存到data_samples中
            data_samples[i].set_field(SeriesData(data=i_seg_logits), 'seg_logits_L3')
            data_samples[i].set_field(SeriesData(data=i_seg_pred), 'pred_sem_seg_L3')
            
        return data_samples

    @torch.inference_mode()
    def slide_inference(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """使用重叠的滑动窗口进行推理。
        
        Args:
            inputs (Tensor): 输入张量，形状为(N, C, Z, Y, X)
            data_samples (Sequence[BaseDataElement], optional): 数据样本
            
        Returns:
            Tensor: 分割结果的logits
        """
        # 获取滑动窗口参数
        assert self.inference_PatchSize is not None and self.inference_PatchStride is not None, \
            f"滑动窗口采样必须指定inference_PatchSize({self.inference_PatchSize})和inference_PatchStride({self.inference_PatchStride})"
        z_stride, _, _ = self.inference_PatchStride
        z_crop, _, _ = self.inference_PatchSize
        batch_size, _, z_img, _, _ = inputs.size()
        
        # 计算网格数
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1
        # 准备结果累加矩阵，根据指定的设备创建
        input_device = inputs.device
        accumulate_device = torch.device(self.inference_PatchAccumulateDevice)
        
        # 创建累加矩阵和计数矩阵在指定的设备上
        preds = torch.zeros(
            size=(batch_size, z_img),
            dtype=torch.float32,
            device=accumulate_device
        )
        count_mat = torch.zeros(
            size=(batch_size, z_img),
            dtype=torch.float32,
            device=accumulate_device
        )
        
        # 滑动窗口推理
        for z_idx in range(z_grids):
            z1 = z_idx * z_stride
            z2 = min(z1 + z_crop, z_img)
            z1 = max(z2 - z_crop, 0)
            
            # 截取patch
            crop_vol = inputs[:, :, z1:z2, ...]
            # 推理
            crop_seg_logit = self._forward(crop_vol) # [B, Z]
            
            # 将结果移到累加设备上并累加
            crop_seg_logit_on_device = crop_seg_logit.to(accumulate_device)
            preds[:, z1:z2]+= crop_seg_logit_on_device
            count_mat[:, z1:z2] += 1
        
        assert torch.all(count_mat > 0), "存在未被滑动窗口覆盖的区域"
        seg_logits = preds / count_mat
        return seg_logits.to(input_device)[:, None, :] # [B, 1, Z]


""" ----- Utils ----- """

@deprecated("The older inferencer is too complex.")
class SarcopeniaL3Inferencer(Inferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: SarcopeniaL3Locating
        self.size:list = self.cfg.size
        self.window_left = self.cfg.wl - self.cfg.ww/2
        self.window_right = self.cfg.wl + self.cfg.ww/2
    
    def _resize(self, Volume_arr:np.ndarray):
        return np.stack([cv2.resize(p, self.size[-2:], interpolation=cv2.INTER_NEAREST_EXACT) 
                         for p in Volume_arr])

    @torch.inference_mode()
    def Inference_FromTensor(self, image_tensor:torch.Tensor):
        assert image_tensor.ndim == 5, f"输入图像必须是5维的，但得到的是 {image_tensor.shape}。"
        if self.fp16:
            image_tensor = image_tensor.half()
        L3_logits = self.model.slide_inference(image_tensor.cuda(), batch_img_metas=[{"img_shape": image_tensor.shape[2:]}])
        L3_pred = torch.sigmoid(L3_logits) > self.model.decode_head.threshold
        return L3_pred # [B, Z]

    def Inference_FromNDArray(self, image_array:np.ndarray) -> np.ndarray:
        # image_array: [Z, Y, X]
        self.model: SarcopeniaL3Locating
        assert image_array.ndim == 3, f"输入图像必须是3维的，但得到的是 {image_array.shape}。"
        
        image_array = self._resize(image_array)
        image_array = (np.clip(image_array, self.window_left, self.window_right) - self.window_left) / self.cfg.ww
        image_tensor = torch.from_numpy(image_array.astype(np.float32))
        L3_pred = self.Inference_FromTensor(image_tensor[None,None]) # [B, C, Z, Y, X]
        return L3_pred[0].cpu().numpy() # [Z]


class SarcopeniaLocateInferencerLite(Inferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: L3LocatingLite
        self.thr:float = self.model.binary_segment_threshold
        self.size:list = self.cfg.size
        self.window_left = self.cfg.wl - self.cfg.ww/2
        self.window_right = self.cfg.wl + self.cfg.ww/2
    
    def _preprocess(self, Volume_arr:np.ndarray):
        Volume_arr = (np.clip(Volume_arr, self.window_left, self.window_right) - self.window_left) / self.cfg.ww
        Volume_arr = np.stack([cv2.resize(p, self.size[-2:], interpolation=cv2.INTER_NEAREST_EXACT) 
                               for p in Volume_arr])
        return Volume_arr

    @torch.inference_mode()
    def Inference_FromTensor(self, image_tensor:torch.Tensor):
        assert image_tensor.ndim == 5, f"输入图像必须是5维的，但得到的是 {image_tensor.shape}。"
        if self.fp16:
            image_tensor = image_tensor.half()
        L3_logits = self.model.slide_inference(image_tensor.cuda())
        L3_pred = torch.sigmoid(L3_logits) > self.thr
        return L3_pred # [B, Z]

    def Inference_FromNDArray(self, image_array:np.ndarray) -> np.ndarray:
        # image_array: [Z, Y, X]
        assert image_array.ndim == 3, f"输入图像必须是3维的，但得到的是 {image_array.shape}。"
        image_array = self._preprocess(image_array)
        image_tensor = torch.from_numpy(image_array.astype(np.float32))
        L3_pred = self.Inference_FromTensor(image_tensor[None,None]) # [B, C, Z, Y, X]
        return L3_pred[0][0].cpu().numpy() # [Z]
