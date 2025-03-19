import pdb
from io import BytesIO
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from mmcv.transforms import BaseTransform
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from mmengine.evaluator.metric import BaseMetric
from mmengine.hooks import Hook
from mmengine.runner import Runner
from ..mm.mmseg_Dev3D import BaseDecodeHead_3D, Seg3DDataSample, PixelShuffle3D, EncoderDecoder_3D


class L3LocationDecoder(BaseDecodeHead_3D):
    def __init__(self, 
                 embed_dims:list[int], 
                 Z_lengths:list[int], 
                 threshold:float=0.3, 
                 loss_weight:float=1., 
                 use_checkpoint:bool=False,
                 pixel_shuffle:int|None=None,
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
        
        # NOTE Detach from the encoder, not influencing the encoder, improve stability.
        feat = x[-1].detach() # [B, C, Z, H, W]
        
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
                List of multi-level img features.
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
        hit = (z_results > self.threshold) == (foreground_Zs > 0)
        
        return {"loss_L3": loss * self.loss_weight,
                "acc_L3": torch.mean(hit.float())}
    
    def predict(self, inputs:list[Tensor], data_samples:list[BaseDataElement]) -> Tensor:
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


class L3_VisHook(Hook):
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


class gen_L3_label(BaseTransform):
    def transform(self, results:dict):
        results['gt_L3'] = np.any(results['gt_seg_map'], axis=(1,2))
        results['seg_fields'].append('gt_L3')
        return results


class SarcopeniaSegmentorWithL3Locating(EncoderDecoder_3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.with_auxiliary_head, "The model should have auxiliary head, it will locate L3."

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
                    self.auxiliary_head: L3LocationDecoder
                    L3_location = self.auxiliary_head.predict(encoder_out, batch_img_metas)[:, None, :, None, None] # [B, Z]
                    seg_logits_foreground = (seg_logits*L3_location).to(accu_device, non_blocking=False)

                    preds[:, :, z1:z2, y1:y2, x1:x2] += seg_logits_foreground
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1

        assert torch.all(count_mat != 0), "The count_mat should not be zero"
        seg_logits = preds / count_mat
        return seg_logits