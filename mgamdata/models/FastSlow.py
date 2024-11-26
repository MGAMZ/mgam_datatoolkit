from abc import abstractmethod
from functools import partial
import os
import pdb
from typing_extensions import Literal

import torch
from torch.nn import PixelUnshuffle as PixelUnshuffle2D

from mmengine.model.base_module import BaseModule
from mmengine.dist import all_gather, get_rank
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup.base import BaseSelfSupervisor
from mmpretrain.models.selfsup.mocov3 import CosineEMA

from mgamdata.mm.mmseg_Dev3D import PixelUnshuffle1D, PixelUnshuffle3D



class MoCoV3Head_WithAcc(BaseModule):
    def __init__(self, 
                 embed_dim:int, 
                 proj_channel:int, 
                 dim:Literal['1d','2d','3d'], 
                 loss: dict, 
                 temperature: float = 1.0) -> None:
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
        if self.dim == '1d':
            proj_conv = torch.nn.Conv1d
            avgpool = partial(torch.nn.AdaptiveAvgPool1d, output_size=(1))
            pus = PixelUnshuffle1D
        elif self.dim == '2d':
            proj_conv = torch.nn.Conv2d
            avgpool = partial(torch.nn.AdaptiveAvgPool2d, output_size=(1,1))
            pus = PixelUnshuffle2D
        elif self.dim == '3d':
            proj_conv = torch.nn.Conv3d
            avgpool = partial(torch.nn.AdaptiveAvgPool3d, output_size=(1,1,1))
            pus = PixelUnshuffle3D
        else:
            raise NotImplementedError(f"Invalid Dim Setting: {self.dim}")
        
        return torch.nn.Sequential(
            pus(downscale_factor=self.down_r), # C_out = factor**dim * C_in
            proj_conv(self.down_r**int(self.dim[0]) * self.embed_dim, self.proj_channel, 1), 
            torch.nn.GELU(), 
            avgpool(), 
            torch.nn.Flatten(start_dim=1), 
        )

    def loss(self, base_out: torch.Tensor,
             momentum_out: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate loss.

        Args:
            base_out (torch.Tensor): [N, C, ...] features from base_encoder.
            momentum_out (torch.Tensor): [N, C, ...] features from momentum_encoder.

        Returns:
            torch.Tensor: The loss tensor.
        """
        # predictor computation
        pred = self.predictor(base_out) # NxC
        target = self.target_proj(base_out) # NxC

        # normalize
        pred = torch.nn.functional.normalize(pred, dim=1)
        target = torch.nn.functional.normalize(target, dim=1)

        # get negative samples
        target = torch.cat(all_gather(target), dim=0)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [pred, target]) / self.temperature

        # generate labels
        batch_size = logits.shape[0]
        labels = (torch.arange(batch_size, dtype=torch.long) +
                  batch_size * get_rank()).to(logits.device)

        loss = self.loss_module(logits, labels)
        return loss, logits, labels


class AutoEncoderSelfSup(BaseSelfSupervisor):
    def __init__(
        self,
        encoder: dict,
        neck: dict | None = None,
        decoder: dict | None = None,
        head: dict | None = None,
        pretrained: str | None = None,
        data_preprocessor: dict | None = None,
        init_cfg: list[dict] | dict | None = None,
        *args,
        **kwargs,
    ) -> None:

        encoder_decoder = torch.nn.Sequential(
            MODELS.build(encoder),
            MODELS.build(neck) if neck is not None else torch.nn.Identity(),
            MODELS.build(decoder) if decoder is not None else torch.nn.Identity(),
        )

        super().__init__(
            backbone=encoder_decoder,
            neck=None,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            *args,
            **kwargs,
        )

    def _get_whole_model(self) -> torch.nn.Module:
        if self.with_neck:
            if self.with_head:
                return torch.nn.Sequential(self.backbone, self.neck, self.head)
            else:
                return torch.nn.Sequential(self.backbone, self.neck)
        else:
            return self.backbone

    @abstractmethod
    def loss(
        self, inputs: list[torch.Tensor], data_samples: list[DataSample], **kwargs
    ) -> dict[str, torch.Tensor]: ...


class AutoEncoder_MoCoV3(AutoEncoderSelfSup):
    def __init__(self, base_momentum: float = 0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_momentum = base_momentum
        self.momentum_encoder = CosineEMA(
            self._get_whole_model(), momentum=base_momentum
        )
    
    @staticmethod
    def calc_acc(logits:torch.Tensor, labels:torch.Tensor) -> torch.Tensor:
        """Calculate the accuracy of the model.
        
        Args:
            logits (torch.Tensor): The output logits, shape (N, C).
            labels (torch.Tensor): The target labels, shape (N).
        
        Returns
            torch.Tensor: The accuracy of the model.
        """
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / labels.shape[0]
        return acc
    
    def loss(
        self, inputs: list[torch.Tensor], data_samples: list[DataSample], **kwargs
    ) -> dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
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
            self.momentum_encoder.update_parameters(self._get_whole_model())

            k1 = self.momentum_encoder(inputs[0])[0]
            k2 = self.momentum_encoder(inputs[1])[0]

        selfsup1 = self.head.loss(q1, k2)
        selfsup2 = self.head.loss(q2, k1)
        
        loss = selfsup1[0] + selfsup2[0]
        acc1 = self.calc_acc(logits=selfsup1[1], labels=selfsup1[2])
        acc2 = self.calc_acc(logits=selfsup2[1], labels=selfsup2[2])
        acc = (acc1 + acc2) / 2

        losses = dict(loss_MoCoV3=loss, acc_MoCoV3=acc)
        return losses
