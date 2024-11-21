import pdb

from torch.nn import L1Loss, MSELoss

from mmengine.model import BaseModule



class PixelReconstructionLoss(BaseModule):
    def __init__(self,
                 criterion="L2",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_name = f"loss_{criterion}"
        self.criterion = L1Loss() if criterion == "L1" else MSELoss()
    
    def forward(self, pred, target, *args, **kwargs):
        return self.criterion(pred.squeeze(), target)

    @property
    def loss_name(self):
        return self._loss_name