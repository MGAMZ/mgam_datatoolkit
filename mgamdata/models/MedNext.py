import pdb
from mmengine.model import BaseModule
from mgamdata.models.mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt

class MM_MedNext(BaseModule):
    def __init__(self, in_channels, embed_dims, kernel_size, **kwargs):
        super().__init__()
        self.backbone = MedNeXt(
            in_channels, embed_dims,
            kernel_size=kernel_size,
            deep_supervision=True,
            dim='2d',
            **kwargs)

    def forward(self, x):
        backbone_out = self.backbone(x)
        return backbone_out



