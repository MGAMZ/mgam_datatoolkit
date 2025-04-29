import pdb
import logging
import copy
from abc import abstractmethod
from typing_extensions import Sequence

import torch
from torch import Tensor

from mmengine.logging import print_log
from mmengine.registry import MODELS
from mmengine.config import ConfigDict
from mmengine.structures import BaseDataElement, PixelData
from mmengine.model import BaseModel

from .mmseg_Dev3D import VolumeData


class mgam_Seg_Lite(BaseModel):
    def __init__(self,
                 backbone:ConfigDict,
                 criterion:ConfigDict|list[ConfigDict],
                 num_classes:int,
                 gt_sem_seg_key:str='gt_sem_seg',
                 use_half:bool=False,
                 binary_segment_threshold:float|None=None,
                 inference_PatchSize:tuple|None=None,
                 inference_PatchStride:tuple|None=None,
                 inference_PatchAccumulateDevice:str='cuda',
                 *args, **kwargs):
        """mgam_Seg_Lite 是一个简化版的分割范式。
        
        与EncoderDecoder2D保持一致的接口，但实现更加简洁。这个类的主要特点：
        1. 简化的前向推理过程，不包含aug_test
        2. decode_head已合并入backbone，backbone直接返回logits
        3. 支持二维滑动窗口推理，当inference_PatchSize和inference_PatchStride被指定时启用
        
        Args:
            backbone (ConfigDict): 主干网络的配置，包含已合并的decode_head。这个主干网络应当直接输出最终的分割logits。
            criterion (ConfigDict): 用于计算损失的标准，通常是Dice或交叉熵损失等。
            gt_sem_seg_key (str): ground truth分割掩码的键名，默认为'gt_sem_seg'。
            use_half (bool): 是否使用半精度模型，默认为False。
            binary_segment_threshold (float | None): 二分类分割的阈值。如果模型输出是单通道 (二分类)，则此参数必须提供；若模型输出是多通道(多分类)，则此参数必须为None。
            inference_PatchSize (tuple | None): 推理时滑动窗口的大小，如果为None，则不使用滑动窗口推理。默认为None。
            inference_PatchStride (tuple | None): 推理时滑动窗口的步长，如果为None，则不使用滑动窗口推理。默认为None。
            inference_PatchAccumulateDevice (str): 推理时滑动窗口结果累加矩阵的存储位置，可以是'cpu'或'cuda'。当处理大图像数据时，选择'cpu'可以避免GPU内存不足。默认为'cuda'。
        """
        super().__init__(*args, **kwargs)
        self.backbone = MODELS.build(backbone)
        self.criterion = [MODELS.build(c) for c in criterion] if isinstance(criterion, list) else [MODELS.build(criterion)]
        self.num_classes = num_classes
        self.gt_sem_seg_key = gt_sem_seg_key
        self.use_half = use_half
        self.binary_segment_threshold = binary_segment_threshold
        self.inference_PatchSize = inference_PatchSize
        self.inference_PatchStride = inference_PatchStride
        self.inference_PatchAccumulateDevice = inference_PatchAccumulateDevice
        
        if use_half:
            self.half()

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

    @abstractmethod
    def loss(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]) -> dict:
        ...
    
    @abstractmethod
    def predict(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Sequence[BaseDataElement]:
        ...
    
    @abstractmethod
    def _forward(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        ...


class mgam_Seg2D_Lite(mgam_Seg_Lite):
    def loss(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]) -> dict:
        """Calculate losses from a batch of inputs and data samples.
        
        Args:
            inputs (Tensor): The input tensor with shape (N, C, H, W)
            data_samples (Sequence[BaseDataElement]): The seg data samples
            
        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # 前向传播，获取预测结果
        seg_logits = self._forward(inputs, data_samples)
        
        # 从data_samples中获取ground truth
        gt_segs = []
        for data_sample in data_samples:
            gt_segs.append(data_sample.get(self.gt_sem_seg_key).data)
        gt_segs = torch.stack(gt_segs, dim=0).squeeze(1)  # [N, H, W]
        
        return {'loss_' + cri.__class__.__name__: cri(seg_logits, gt_segs) 
                for cri in self.criterion}

    def predict(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Sequence[BaseDataElement]:
        """Predict results from a batch of inputs and data samples.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, H, W).
            data_samples (Sequence[BaseDataElement], optional): The seg data samples.
                It usually includes information such as `metainfo`.
                
        Returns:
            Sequence[BaseDataElement]: Segmentation results of the input images.
                Each SegDataSample usually contains:
                - pred_sem_seg (PixelData): Prediction of semantic segmentation.
                - seg_logits (PixelData): Predicted logits of semantic segmentation.
        """
        # 前向传播
        seg_logits = self.inference(inputs, data_samples) # [N, C, H, W]
        
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
            i_seg_logits = seg_logits[i] # [C, H, W]
            
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
            data_samples[i].seg_logits = PixelData(data=i_seg_logits)
            data_samples[i].pred_sem_seg = PixelData(data=i_seg_pred)
        
        return data_samples

    def _forward(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): The input tensor with shape (N, C, H, W).
            data_samples (Sequence[BaseDataElement], optional): The seg data samples.
            
        Returns:
            Tensor: Output tensor from backbone
        """
        return self.backbone(inputs)

    @torch.inference_mode()
    def inference(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        """执行推理，支持滑动窗口或整体推理。
        
        Args:
            inputs (Tensor): 输入张量，形状为(N, C, H, W)
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
            inputs (Tensor): 输入张量，形状为(N, C, H, W)
            data_samples (Sequence[BaseDataElement], optional): 数据样本
            
        Returns:
            Tensor: 分割结果的logits
        """
        # 获取滑动窗口参数
        assert self.inference_PatchSize is not None and self.inference_PatchStride is not None, \
            f"滑动窗口采样必须指定inference_PatchSize({self.inference_PatchSize})和inference_PatchStride({self.inference_PatchStride})"
        h_stride, w_stride = self.inference_PatchStride
        h_crop, w_crop = self.inference_PatchSize
        batch_size, _, h_img, w_img = inputs.size()
        
        # 获取输出通道数（类别数）
        with torch.no_grad():
            temp_output = self._forward(inputs[:, :, :min(h_crop, h_img), 
                                               :min(w_crop, w_img)])
            out_channels = temp_output.size(1)
        
        # 计算网格数
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        
        accumulate_device = torch.device(self.inference_PatchAccumulateDevice)
        
        preds = torch.zeros(
            size=(batch_size, out_channels, h_img, w_img),
            dtype=torch.float16,
            device=accumulate_device
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, h_img, w_img),
            dtype=torch.uint8,
            device=accumulate_device
        )
        
        # 滑动窗口推理
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                h1 = h_idx * h_stride
                w1 = w_idx * w_stride
                h2 = min(h1 + h_crop, h_img)
                w2 = min(w1 + w_crop, w_img)
                h1 = max(h2 - h_crop, 0)
                w1 = max(w2 - w_crop, 0)
                
                # 截取patch
                crop_img = inputs[:, :, h1:h2, w1:w2]
                
                # 推理
                crop_seg_logit = self._forward(crop_img)
                
                # 将结果移到累加设备上并累加
                crop_seg_logit_on_device = crop_seg_logit.to(accumulate_device)
                preds[:, :, h1:h2, w1:w2] += crop_seg_logit_on_device
                count_mat[:, :, h1:h2, w1:w2] += 1
        
        assert torch.all(count_mat > 0), "存在未被滑动窗口覆盖的区域"
        seg_logits = preds / count_mat
        
        return seg_logits


class mgam_Seg3D_Lite(mgam_Seg_Lite):
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
            gt_segs.append(data_sample.get(self.gt_sem_seg_key).data)
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
        return self.backbone(inputs)

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
        
        # 计算网格数
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(y_img - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(x_img - x_crop + x_stride - 1, 0) // x_stride + 1
        
        # 准备结果累加矩阵，根据指定的设备创建
        accumulate_device = torch.device(self.inference_PatchAccumulateDevice)
        
        # 创建累加矩阵和计数矩阵在指定的设备上
        preds = torch.zeros(
            size=(batch_size, self.num_classes, z_img, y_img, x_img),
            dtype=torch.float16,
            device=accumulate_device
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, z_img, y_img, x_img),
            dtype=torch.uint8,
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
                    
                    # 推理
                    crop_seg_logit = self._forward(inputs[:, :, z1:z2, y1:y2, x1:x2])
                    # 累加
                    preds[:, :, z1:z2, y1:y2, x1:x2] += crop_seg_logit.to(accumulate_device)
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1
        
        # 确保没有未覆盖区域
        assert torch.all(count_mat > 0), "存在未被滑动窗口覆盖的区域"
        # 计算平均值
        seg_logits = (preds / count_mat).to(dtype=torch.float16)
        
        return seg_logits


class MomentumAvgModel(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 100,
                 interval: int = 1,
                 device: torch.device|None = None,
                 update_buffers: bool = False) -> None:
        super().__init__()
        
        # 检查分布式环境
        self.is_distributed = hasattr(model, 'module')
        self.is_deepspeed = hasattr(model, 'module') and hasattr(model.module, 'deepspeed')
        
        # DeepSpeed环境下获取完整模型
        if self.is_deepspeed:
            with model.module.summon_full_params():
                self.module = copy.deepcopy(model.module).requires_grad_(False)
        else:
            target_model = model.module if self.is_distributed else model
            self.module = copy.deepcopy(target_model).requires_grad_(False)
            
        self.interval = interval
        if device is not None:
            self.module = self.module.to(device)
            
        self.register_buffer('steps', torch.tensor(0, dtype=torch.long, device=device))
                           
        self.update_buffers = update_buffers
        if update_buffers:
            state_dict = self.module.state_dict()
            self.avg_parameters = {
                k: v for k, v in state_dict.items() 
                if v.numel() > 0
            }
        else:
            params = dict(self.module.named_parameters())
            self.avg_parameters = {k: v for k, v in params.items() 
                                   if v.numel() > 0}
            
        # 动量参数检查
        assert 0.0 < momentum < 1.0, f'momentum must be in range (0.0, 1.0) but got {momentum}'
        if momentum > 0.5:
            print_log('The value of momentum in EMA is usually a small number,'
                      'which is different from the conventional notion of '
                      f'momentum but got {momentum}. Please make sure the '
                      f'value is correct.',
                      logger='current', 
                      level=logging.WARNING)
        self.momentum = momentum
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)

    def _get_current_param(self):
        if self.update_buffers:
            return self.module.state_dict()
        else:
            return dict(self.module.named_parameters())
    
    def update_parameters(self, model: torch.nn.Module) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = (
            model.state_dict()
            if self.update_buffers else dict(model.named_parameters()))
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:  # type: ignore
            for k, p_avg in self.avg_parameters.items():
                # NOTE handle deepspeed model shred issue, p_avg may be empty here.
                if p_avg.dtype.is_floating_point and p_avg.shape==src_parameters[k].data.shape:
                    device = p_avg.device
                    self.avg_func(p_avg.data,
                                  src_parameters[k].data.to(device),
                                  self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data.to(b_avg.device))
        self.steps += 1  # type: ignore

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using the linear
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = max(self.momentum,
                       self.gamma / (self.gamma + self.steps.item()))
        averaged_param.lerp_(source_param, momentum)
