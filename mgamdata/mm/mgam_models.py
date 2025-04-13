from collections import defaultdict
import os
import pdb
from abc import abstractmethod
from tqdm import tqdm
from typing_extensions import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.registry import MODELS
from mmengine.config import ConfigDict
from mmengine.structures import BaseDataElement
from mmengine.model import BaseModel

from .mmseg_Dev3D import VolumeData



class mgam_Seg3D_Lite(BaseModel):
    def __init__(self,
                 backbone:ConfigDict,
                 criterion:ConfigDict|list[ConfigDict],
                 binary_segment_threshold:float|None=None,
                 inference_PatchSize:tuple|None=None,
                 inference_PatchStride:tuple|None=None,
                 inference_PatchAccumulateDevice:str='cuda',
                 *args, **kwargs):
        """mgam_Seg3D_Lite 是一个简化版的3D分割模型实现。
        
        与EncoderDecoder3D保持一致的接口，但实现更加简洁。这个类的主要特点：
        1. 简化的前向推理过程，不包含aug_test
        2. decode_head已合并入backbone，backbone直接返回logits
        3. 支持三维滑动窗口推理，当inference_PatchSize和inference_PatchStride被指定时启用
        
        Args:
            backbone (ConfigDict): 主干网络的配置，包含已合并的decode_head。
                这个主干网络应当直接输出最终的分割logits。
            criterion (ConfigDict): 用于计算损失的标准，通常是Dice或交叉熵损失等。
            binary_segment_threshold (float | None): 二分类分割的阈值。如果模型输出是单通道
                (二分类)，则此参数必须提供；若模型输出是多通道(多分类)，则此参数必须为None。
                默认为0.5。
            inference_PatchSize (tuple | None): 推理时滑动窗口的大小 (z, y, x)，
                如果为None，则不使用滑动窗口推理。默认为None。
            inference_PatchStride (tuple | None): 推理时滑动窗口的步长 (z, y, x)，
                如果为None，则不使用滑动窗口推理。默认为None。
            inference_PatchAccumulateDevice (str): 推理时滑动窗口结果累加矩阵的存储位置，
                可以是'cpu'或'cuda'。当处理大体积数据时，选择'cpu'可以避免GPU内存不足。
                默认为'cuda'。
        """
        
        super().__init__(*args, **kwargs)
        self.backbone = MODELS.build(backbone)
        self.criterion = [MODELS.build(c) for c in criterion] if isinstance(criterion, list) else [MODELS.build(criterion)]
        self.binary_segment_threshold = binary_segment_threshold
        self.inference_PatchSize = inference_PatchSize
        self.inference_PatchStride = inference_PatchStride
        self.inference_PatchAccumulateDevice = inference_PatchAccumulateDevice

    def forward(self,
                inputs: Tensor,
                data_samples:Sequence[BaseDataElement]|None=None,
                mode:str='tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]) -> dict:
        """Calculate losses from a batch of inputs and data samples.
        
        Args:
            inputs (Tensor): The input tensor with shape (N, C, Z, Y, X)
            data_samples (Sequence[BaseDataElement]): The seg data samples
            
        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # 前向传播，获取预测结果
        seg_logits = self._forward(inputs, data_samples)
        
        # 从data_samples中获取ground truth
        gt_segs = []
        for data_sample in data_samples:
            gt_segs.append(data_sample.gt_sem_seg.data)
        gt_segs = torch.stack(gt_segs, dim=0).squeeze(1)  # [N, Z, Y, X]
        
        return {'loss_' + cri.__class__.__name__: cri(seg_logits, gt_segs) 
                for cri in self.criterion}

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
            data_samples[i].seg_logits = VolumeData(**{"data": i_seg_logits})
            data_samples[i].pred_sem_seg = VolumeData(**{"data": i_seg_pred})
            
        return data_samples

    def _forward(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, Z, Y, X).
            data_samples (Sequence[BaseDataElement], optional): The seg data samples.
            
        Returns:
            Tensor: Output tensor from backbone
        """

        x = self.backbone(inputs) # [N, C, Z, Y, X]
        N, C, Z, Y, X = x.shape
        if C == 1:
            return F.sigmoid(x)
        elif C > 1:
            return F.softmax(x, dim=1)
        else:
            raise ValueError(f"输出通道数({C})必须大于0")

    @torch.inference_mode()
    def inference(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """执行推理，支持滑动窗口或整体推理。
        
        Args:
            inputs (Tensor): 输入张量，形状为(N, C, Z, Y, X)
            data_samples (Sequence[BaseDataElement], optional): 数据样本
            
        Returns:
            Tensor: 分割结果的logits
        """
        # 检查是否需要滑动窗口推理
        if self.inference_PatchSize is not None and self.inference_PatchStride is not None:
            seg_logits = self.slide_inference(inputs, data_samples)
        else:
            # 整体推理
            seg_logits = self._forward(inputs, data_samples)
            
        return seg_logits

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
        z_stride, y_stride, x_stride = self.inference_PatchStride
        z_crop, y_crop, x_crop = self.inference_PatchSize
        batch_size, _, z_img, y_img, x_img = inputs.size()
        
        # 获取输出通道数（类别数）
        with torch.no_grad():
            temp_output = self._forward(inputs[:, :, :min(z_crop, z_img), 
                                               :min(y_crop, y_img), 
                                               :min(x_crop, x_img)])
            out_channels = temp_output.size(1)
        
        # 计算网格数
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(y_img - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(x_img - x_crop + x_stride - 1, 0) // x_stride + 1
        
        # 准备结果累加矩阵，根据指定的设备创建
        input_device = inputs.device
        accumulate_device = torch.device(self.inference_PatchAccumulateDevice)
        
        # 创建累加矩阵和计数矩阵在指定的设备上
        preds = torch.zeros(
            size=(batch_size, out_channels, z_img, y_img, x_img),
            dtype=torch.float32,
            device=accumulate_device
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, z_img, y_img, x_img),
            dtype=torch.float32,
            device=accumulate_device
        )
        
        # 滑动窗口推理
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
                    
                    # 截取patch
                    crop_vol = inputs[:, :, z1:z2, y1:y2, x1:x2]
                    
                    # 推理
                    crop_seg_logit = self._forward(crop_vol)
                    
                    # 将结果移到累加设备上并累加
                    crop_seg_logit_on_device = crop_seg_logit.to(accumulate_device)
                    preds[:, :, z1:z2, y1:y2, x1:x2] += crop_seg_logit_on_device
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1
        
        # 确保没有未覆盖区域
        assert torch.all(count_mat > 0), "存在未被滑动窗口覆盖的区域"
        
        # 计算平均值
        seg_logits = preds / count_mat
        
        # 将结果移回输入设备
        seg_logits = seg_logits.to(input_device)
        
        return seg_logits
