import pdb

import torch
import segmentation_models_pytorch as smp
from timm.models._pretrained import PretrainedCfg
from timm import create_model

class SMP_Segformer(torch.nn.Module):
    def __init__(self,
                 encoder_name:str,
                 in_channels:int,
                 num_classes:int,
                 encoder_depth:int=5,
                 encoder_weights:str='imagenet',
                 decoder_segmentation_channels=256,
                 timm_ckpt_local_path:str|None=None,
                 **kwargs):
        super().__init__()
        self.efficient_vit = smp.Segformer(
            in_channels=in_channels,
            classes=num_classes,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_segmentation_channels=decoder_segmentation_channels,
            pretrained_cfg_overlay=dict(file=timm_ckpt_local_path) if timm_ckpt_local_path else {},
            **kwargs)
    
    def forward(self, x:torch.Tensor):
        return self.efficient_vit(x)


if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    model = SMP_Segformer(
        encoder_name="tu-efficientvit_b3",
        in_channels=3,
        num_classes=1,
        pretrained_cfg_overlay=dict(file='/fileser51/zhangyiqin.sx/timm_ckpts/efficientvit_b3.r224_in1k.safetensors'),
        cache_dir='/fileser51/zhangyiqin.sx/timm_ckpts/'
    )
    y = model(x)
    print(y.shape)