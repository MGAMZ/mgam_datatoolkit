
from mmengine.model import BaseModule

from .SwinUMamba import VSSMEncoder, UNetResDecoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

class MM_SwinUMamba_backbone(BaseModule):
    def __init__(self, 
                 in_chans=3, 
                 patch_size=4, 
                 depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], 
                 d_state=16):
        super().__init__()
        self.model = VSSMEncoder(patch_size, in_chans, depths, dims, d_state)
    
    def forward(self, x):
        return self.model(x)


class MM_SwinUMamba_decoder(BaseDecodeHead):
    def __init__(self, 
                 num_classes, 
                 in_channels=[96, 192, 384, 768], 
                 d_state=16,
                 *args, **kwargs):
        super().__init__(input_transform='multiple_select', 
                         in_channels=in_channels, 
                         num_classes=num_classes, 
                         init_cfg=None,
                         *args, **kwargs)
        self.model = UNetResDecoder(num_classes, features_per_stage=in_channels, d_state=d_state)
        del self.conv_seg # not using mmseg built-in cls seg conv

    def forward(self, x):
        return self.model(x)

