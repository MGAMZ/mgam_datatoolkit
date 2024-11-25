import os
import pdb

import torch
from mmengine.model.base_module import BaseModule
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup.base import BaseSelfSupervisor
from mmpretrain.models.selfsup.mocov3 import CosineEMA


class WholeNetwork_MoCoV3(BaseSelfSupervisor):
    def __init__(self,
        backbone: dict,
        neck: dict|None=None,
        head: dict|None=None,
        base_momentum: float = 0.01,
        pretrained: str|None = None,
        data_preprocessor: dict|None = None,
        init_cfg: list[dict]|dict|None = None,
        *args, **kwargs
    ) -> None:
        
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            *args, **kwargs
        )
        # create momentum model
        self.momentum_encoder = CosineEMA(
            self._get_whole_model(), momentum=base_momentum)

    def _get_whole_model(self):
        if self.with_neck:
            if self.with_head:
                return torch.nn.Sequential(self.backbone, self.neck, self.head)
            else:
                return torch.nn.Sequential(self.backbone, self.neck)
        else:
            return self.backbone

    def loss(self, inputs: list[torch.Tensor], data_samples: list[DataSample],
             **kwargs) -> dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        assert isinstance(inputs, list)
        self.backbone:BaseModule
        self.neck:BaseModule
        self.head:BaseModule
        
        view_1 = inputs[0]
        view_2 = inputs[1]

        x1 = self.backbone(view_1)
        x2 = self.backbone(view_2)
        if self.with_neck:
            x1 = self.neck(x1)[0]
            x2 = self.neck(x2)[0]
        if self.with_head:
            x1 = self.head(x1)[0]
            x2 = self.head(x2)[0]
        
        q1 = x1
        q2 = x2

        # compute key features, [N, C] each, no gradient
        with torch.no_grad():
            # update momentum encoder
            self.momentum_encoder.update_parameters(self._get_whole_model())

            k1 = self.momentum_encoder(view_1)[0]
            k2 = self.momentum_encoder(view_2)[0]

        loss = self.head.loss(q1, k2) + self.head.loss(q2, k1)

        losses = dict(loss_MoCoV3=loss)
        return losses
