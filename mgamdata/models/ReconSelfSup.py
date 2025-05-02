import os
import pdb
from typing_extensions import Literal, Sequence

import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor

from mmcv.transforms import BaseTransform
from mmengine.evaluator.metric import BaseMetric
from mmengine.structures import BaseDataElement
from mmengine.model import BaseModule
from mmengine.dist import master_only

from ..mm.mmeng_PlugIn import GeneralViser
from .SelfSup import AutoEncoderSelfSup, VoxelData



class ReconDataSample(BaseDataElement):
    def set_gt_data(self, value:Tensor):
        self.set_field(value, 'gt_data', dtype=VoxelData)

    def set_mask(self, value:Tensor):
        self.set_field(value, 'erase_mask', dtype=VoxelData)
    
    def set_pred_data(self, value:Tensor):
        self.set_field(value, 'pred_data', dtype=VoxelData)


class PackReconInput(BaseTransform):
    def transform(self, results:dict):
        inputs = torch.from_numpy(results['img'])
        datasample = ReconDataSample(
            erased_data=VoxelData(data=results['img']),
            gt_data=VoxelData(data=torch.from_numpy(results['ori_img'])),
            erase_mask=VoxelData(data=torch.from_numpy(results['erase_mask'])),
            metainfo={"sample_file_path": results['img_path'],}
        )
        
        return {
            "inputs": inputs,
            "data_samples": datasample,
        }


class ReconHead(BaseModule):
    def __init__(
        self,
        model_out_channels: int,
        recon_channels: int,
        dim: Literal["1d", "2d", "3d"],
        loss_type: Literal["L1", "L2"] = "L1",
    ):
        super().__init__()
        self.model_out_channels = model_out_channels
        self.recon_channels = recon_channels
        self.loss_type = loss_type
        self.dim = dim
        self.criterion = (
            nn.L1Loss(reduction="none")
            if loss_type == "L1"
            else nn.MSELoss(reduction="none"))
        self.conv_proj = eval(f"nn.Conv{dim}")(
            model_out_channels, recon_channels, 1
        )

    def loss(self, recon: Tensor, ori: Tensor, mask:Tensor|None=None) -> dict[str, Tensor]:
        proj = self(recon)
        loss = self.criterion(proj, ori)
        if mask is not None:
            loss = loss * mask
        return {f"loss_recon_{self.loss_type}": loss.mean()}

    def forward(self, recon:Tensor) -> Tensor:
        return self.conv_proj(recon)


class Reconstructor(AutoEncoderSelfSup):
    head: ReconHead
    
    def __init__(self, recon_channels:int, test_cfg:dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recon_channels = recon_channels
        self.test_cfg = test_cfg
    
    def _stack_datasamples(self, data_samples: list[ReconDataSample]) -> tuple[Tensor, Tensor]:
        ori = torch.stack([sample.gt_data.data for sample in data_samples])
        mask = torch.stack([sample.erase_mask.data for sample in data_samples])
        return ori, mask
    
    def loss(
        self, 
        inputs: list[Tensor], 
        data_samples: list[ReconDataSample]
    ) -> dict[str, Tensor]:
        
        recon = self.whole_model_(inputs)
        ori, mask = self._stack_datasamples(data_samples)
        selfsup_loss = self.head.loss(recon[0], ori, mask)
        return selfsup_loss

    def slide_inference(self, inputs: Tensor) -> Tensor:
        """Inference by sliding-window with overlap, copy from `EncoderDecoder3D`.

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
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(y_img - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(x_img - x_crop + x_stride - 1, 0) // x_stride + 1
        preds = torch.zeros(
            size=(batch_size, self.recon_channels, z_img, y_img, x_img),
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
                    crop_vol = inputs[:, :, z1:z2, y1:y2, x1:x2]  # [N, C, Z, Y, X]
                    
                    # NOTE WARNING:
                    # Setting `non_blocking=True` WILL CAUSE:
                    # Invalid pred_seg_logit accumulation on X axis.
                    crop_seg_logit = self.whole_model_(crop_vol)[0]
                    reconed = self.head(crop_seg_logit)
                    
                    preds[:, :, z1:z2, y1:y2, x1:x2] += reconed.to(accu_device, non_blocking=False)
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1

        assert torch.all(count_mat != 0), "The count_mat should not be zero"
        preds /= count_mat
        return preds

    def whole_inference(self, inputs: Tensor) -> Tensor:
        recon_feat = self.whole_model_(inputs)[0]
        reconed = self.head(recon_feat)
        return reconed

    @torch.inference_mode()
    def predict(
        self, 
        inputs: Tensor, 
        data_samples: list[ReconDataSample]
    ) -> list[ReconDataSample]:
        
        if self.test_cfg["mode"] == "whole":
            reconed = self.whole_inference(inputs)
        elif self.test_cfg["mode"] == "slide":
            reconed = self.slide_inference(inputs)
        
        for i, sample in enumerate(data_samples):
            sample.set_pred_data(VoxelData(data=reconed[i]))
        
        return data_samples

    def forward(self,
                inputs: Tensor,
                data_samples: list[ReconDataSample],
                mode: str = 'tensor'
    ):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')


class ReconMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.L1 = nn.L1Loss(reduction="mean")
        self.eps = 1e-6
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = sample["pred_data"]["data"]
            gt = sample["gt_data"]["data"]
            self.results.append({
                "L1": self.L1(pred, gt).cpu().numpy(),
                "mape": torch.mean(torch.abs(pred - gt) / (gt + self.eps)).cpu().numpy()
            })
    
    def compute_metrics(self, results: list[dict]) -> dict:
        L1 = sum([r['L1'] for r in results]) / len(results)
        mape = sum([r['mape'] for r in results]) / len(results)
        return {"mae": L1, "mape": mape}


class ReconViser(GeneralViser):
    @master_only
    def add_datasample(self, data_sample:ReconDataSample, step:int|None=None):
        input_data = data_sample.erased_data.data.mean(axis=0)
        gt_data = data_sample.gt_data.data.detach().cpu().numpy().mean(axis=0)
        pred_data = data_sample.pred_data.data.detach().cpu().numpy().mean(axis=0)
        z_mid = gt_data.shape[0] // 2
        y_mid = gt_data.shape[1] // 2
        x_mid = gt_data.shape[2] // 2
        fig, axes = plt.subplots(3, 3, figsize=(10, 6))
        
        axes[0,0].imshow(input_data[z_mid, ...], cmap="gray")
        axes[0,1].imshow(input_data[:, y_mid, :], cmap="gray")
        axes[0,2].imshow(input_data[:, :, x_mid], cmap="gray")
        axes[1,0].imshow(gt_data[z_mid, ...], cmap="gray")
        axes[1,1].imshow(gt_data[:, y_mid, :], cmap="gray")
        axes[1,2].imshow(gt_data[:, :, x_mid], cmap="gray")
        axes[2,0].imshow(pred_data[z_mid, ...], cmap="gray")
        axes[2,1].imshow(pred_data[:, y_mid, :], cmap="gray")
        axes[2,2].imshow(pred_data[:, :, x_mid], cmap="gray")
        
        axes[0,0].set_title("XY")
        axes[0,1].set_title("YZ")
        axes[0,2].set_title("XZ")
        axes[0,0].set_ylabel("Input")
        axes[1,0].set_ylabel("GT")
        axes[2,0].set_ylabel("Pred")
        
        plt.tight_layout()
        fig_array = self._plt2array(fig)
        plt.close(fig)
        
        dir_name = os.path.basename(os.path.dirname(data_sample.sample_file_path))
        self.add_image(name=dir_name, image=fig_array, step=step)


