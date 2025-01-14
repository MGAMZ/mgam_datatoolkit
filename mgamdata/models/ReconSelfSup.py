import os
import pdb
from functools import partial
from typing_extensions import Literal

import torch
from torch import nn, Tensor
from torch.nn import PixelUnshuffle as PixelUnshuffle2D

from mmengine.registry import MODELS
from mmengine.structures import BaseDataElement
from mmengine.model import BaseModule
from mmengine.dist import all_gather, get_rank
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup.mocov3 import CosineEMA

from ..mm.mmseg_Dev3D import PixelUnshuffle1D, PixelUnshuffle3D
from .SelfSup import AutoEncoderSelfSup, VoxelData



class MoCoV3Head_WithAcc(BaseModule):
    def __init__(
        self,
        embed_dim: int,
        proj_channel: int,
        dim: Literal["1d", "2d", "3d"],
        loss: dict,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_channel = proj_channel
        self.dim = dim
        self.loss_module = MODELS.build(loss)
        self.temperature = temperature
        self.down_r = 4
        self.predictor = self._init_proj()
        self.target_proj = self._init_proj()

    def _init_proj(self):
        if self.dim == "1d":
            proj_conv = nn.Conv1d
            avgpool = partial(nn.AdaptiveAvgPool1d, output_size=(1))
            pus = PixelUnshuffle1D
        elif self.dim == "2d":
            proj_conv = nn.Conv2d
            avgpool = partial(nn.AdaptiveAvgPool2d, output_size=(1, 1))
            pus = PixelUnshuffle2D
        elif self.dim == "3d":
            proj_conv = nn.Conv3d
            avgpool = partial(nn.AdaptiveAvgPool3d, output_size=(1, 1, 1))
            pus = PixelUnshuffle3D
        else:
            raise NotImplementedError(f"Invalid Dim Setting: {self.dim}")

        return nn.Sequential(
            pus(downscale_factor=self.down_r),  # C_out = factor**dim * C_in
            proj_conv(
                self.down_r ** int(self.dim[0]) * self.embed_dim, self.proj_channel, 1
            ),
            nn.GELU(),
            avgpool(),
            nn.Flatten(start_dim=1),
        )

    def loss(
        self, base_out: Tensor, momentum_out: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate loss.

        Args:
            base_out (Tensor): [N, C, ...] features from base_encoder.
            momentum_out (Tensor): [N, C, ...] features from momentum_encoder.

        Returns:
            Tensor: The loss tensor.
        """
        # predictor computation
        pred = self.predictor(base_out)  # NxC
        target = self.target_proj(base_out)  # NxC

        # normalize
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(target, dim=1)

        # get negative samples
        target = torch.cat(all_gather(target), dim=0)

        # Einstein sum is more intuitive
        logits = torch.einsum("nc,mc->nm", [pred, target]) / self.temperature

        """
        使用一个混淆矩阵来表达经过两组不同的变换之后的同batch样本之间的相似度
        理想情况下，模型应当能识别出同样的样本，因此这个矩阵应当是对角线上有较大值，其他地方为较小值
        从分类任务混淆矩阵的角度出发，这代表着样本的gt标签就是它们自身的index
        """

        # generate labels
        batch_size = logits.shape[0]
        labels = (
            torch.arange(batch_size, dtype=torch.long) + batch_size * get_rank()
        ).to(logits.device)

        loss = self.loss_module(logits, labels)
        return loss, logits, labels


class MoCoV3(AutoEncoderSelfSup):
    def __init__(self, base_momentum: float = 0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_momentum = base_momentum
        self.momentum_encoder = CosineEMA(
            self.whole_model_, momentum=base_momentum
        )

    @staticmethod
    def calc_acc(logits: Tensor, labels: Tensor) -> Tensor:
        """Calculate the accuracy of the model.

        Args:
            logits (Tensor): The output logits, shape (N, C).
            labels (Tensor): The target labels, shape (N).

        Returns
            Tensor: The accuracy of the model.
        """
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / labels.shape[0]
        return acc.unsqueeze(0)

    def loss(
        self, inputs: list[Tensor], data_samples: list[DataSample], **kwargs
    ) -> dict[str, Tensor]:
        """The forward function in training.

        Args:
            inputs (List[Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(inputs, list)
        self.backbone: BaseModule
        self.neck: BaseModule
        self.head: BaseModule

        q1 = self.backbone(inputs[0])[0]
        q2 = self.backbone(inputs[1])[0]

        # compute key features, [N, C] each, no gradient
        with torch.no_grad():
            # update momentum encoder
            self.momentum_encoder.update_parameters(self.whole_model_)

            k1 = self.momentum_encoder(inputs[0])[0]
            k2 = self.momentum_encoder(inputs[1])[0]

        selfsup1 = self.head.loss(q1, k2)
        selfsup2 = self.head.loss(q2, k1)

        loss = selfsup1[0] + selfsup2[0]
        acc1 = self.calc_acc(logits=selfsup1[1], labels=selfsup1[2])
        acc2 = self.calc_acc(logits=selfsup2[1], labels=selfsup2[2])
        acc = (acc1 + acc2) / 2
        acc = torch.cat(all_gather(acc)).mean()
        losses = dict(loss_MoCoV3=loss, acc_MoCoV3=acc)
        return losses


class ReconHead(BaseModule):
    def __init__(
        self,
        model_out_channels: int,
        recon_channels: int,
        dim: Literal["1d", "2d", "3d"],
        reduction: str = "mean",
        loss_type: Literal["L1", "L2"] = "L1",
    ):
        super().__init__()
        self.model_out_channels = model_out_channels
        self.recon_channels = recon_channels
        self.loss_type = loss_type
        self.dim = dim
        self.criterion = (
            nn.L1Loss(reduction=reduction)
            if loss_type == "L1"
            else nn.MSELoss(reduction=reduction)
        )
        self.conv_proj = eval(f"nn.Conv{dim}")(
            model_out_channels, recon_channels, 1
        )

    def loss(self, recon: Tensor, ori: Tensor):
        proj = self.conv_proj(recon)
        loss = self.criterion(proj.squeeze(), ori.squeeze())
        return {f"loss_recon_{self.loss_type}": loss, "reconed": proj}


class ReconDataSample(BaseDataElement):
    def set_gt_data(self, value:Tensor):
        self.set_field(value, 'gt_data', dtype=VoxelData)

    def set_mask(self, value:Tensor):
        self.set_field(value, 'mask', dtype=VoxelData)


class Recon_SelfSup(AutoEncoderSelfSup):
    def _stack_datasamples(self, data_samples: list[ReconDataSample]) -> tuple[Tensor, Tensor]:
        ori = torch.stack([sample.gt_data for sample in data_samples])
        return ori
    
    def loss(
        self, 
        inputs: list[Tensor], 
        data_samples: list[ReconDataSample], **kwargs
    ) -> dict[str, Tensor]:
        
        assert isinstance(inputs, list)
        self.backbone: BaseModule
        self.head: BaseModule
        losses = {}
        recon = self.backbone(inputs)
        ori = torch.stack([sample.gt_data for sample in data_samples])
        selfsup_loss = self.head.loss(recon, ori)

        losses.update(selfsup_loss)
        return losses
