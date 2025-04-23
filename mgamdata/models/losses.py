import pdb

import torch
from torch import Tensor
from torch.nn import L1Loss, MSELoss

from mmengine.model import BaseModule


class PixelReconstructionLoss(BaseModule):
    def __init__(
        self,
        criterion="L2",
        use_sigmoid: bool = False,
        reduction="mean",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loss_name = f"loss_{criterion}"
        self.criterion = (
            L1Loss(reduction=reduction)
            if criterion == "L1"
            else MSELoss(reduction=reduction)
        )
        self.use_sigmoid = use_sigmoid

    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        if self.use_sigmoid:
            pred = pred.sigmoid()
        return self.criterion(pred.squeeze(), target.to(pred.dtype))

    @property
    def loss_name(self):
        return self._loss_name


class HingeEmbeddingLoss(BaseModule):
    """Hinge Embedding loss.

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        pos_weight (float): The positive weight for the loss. Defaults to 1.0.
    """

    def __init__(self,
                 num_classes:int, 
                 reduction='mean',
                 loss_weight=1.0,
                 pos_weight=1.0):
        super(HingeEmbeddingLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight
        self.loss = torch.nn.HingeEmbeddingLoss()

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        label[label<0.5] = -1
        label[label>=0.5] = 1
        
        loss_cls = self.loss(cls_score, label)
        if reduction == 'sum':
            loss_cls = loss_cls.sum()
        elif reduction == 'mean':
            loss_cls = loss_cls.mean()
        elif reduction == 'none':
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
        
        return self.loss_weight * loss_cls
