from abc import abstractmethod
from functools import partial
import os
import pdb

import torch
from torch.nn import functional as F

from mmengine.model.base_module import BaseModule
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup.base import BaseSelfSupervisor
from mmpretrain.models.selfsup.mocov3 import CosineEMA


class AutoEncoderSelfSup(BaseSelfSupervisor):
    def __init__(
        self,
        encoder: dict,
        neck: dict | None = None,
        decoder: dict | None = None,
        head: dict | None = None,
        base_momentum: float = 0.01,
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
        # create momentum model
        self.momentum_encoder = CosineEMA(
            self._get_whole_model(), momentum=base_momentum
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

        if q1.ndim == 5:
            avgpool = partial(F.adaptive_avg_pool3d, output_size=(1,1,1))
        elif q1.ndim == 4:
            avgpool = partial(F.adaptive_avg_pool2d, output_size=(1,1))
        elif q1.ndim == 3:
            avgpool = partial(F.adaptive_avg_pool1d, output_size=(1))
        else:
            raise RuntimeError(f"Unsupported ndim {q1.ndim}")

        q1 = avgpool(q1).squeeze()
        q2 = avgpool(q2).squeeze()
        k1 = avgpool(k1).squeeze()
        k2 = avgpool(k2).squeeze()
        loss = self.head.loss(q1, k2) + self.head.loss(q2, k1)

        losses = dict(loss_MoCoV3=loss)
        return losses
