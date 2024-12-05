import pdb

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
        return self.criterion(pred.squeeze(), target)

    @property
    def loss_name(self):
        return self._loss_name
