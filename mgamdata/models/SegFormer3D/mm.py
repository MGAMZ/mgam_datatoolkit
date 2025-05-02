import pdb
from warnings import warn

import torch
from torch import nn
from mmengine.model import BaseModule
from .SegFormer3D import PatchEmbedding, TransformerBlock, cube_root, SegFormerDecoderHead
from ...mm.mmseg_Dev3D import BaseDecodeHead_3D
from ...mm.mgam_models import mgam_Seg3D_Lite

class SegFormer3D_Encoder_MM(BaseModule):
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list = [8, 4, 2, 1],
        embed_dims: list = [64, 128, 320, 512],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [2, 2, 2, 2],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
        freeze: bool = False,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.freeze = freeze

        self.embeds = nn.ModuleList([
            PatchEmbedding(
                in_channel=(in_channels if i == 0 else embed_dims[i-1]),
                embed_dim=embed_dims[i],
                kernel_size=patch_kernel_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
            )
            for i in range(4)
        ])

        self.blocks = []
        self.norms = []
        for i in range(4):
            tf_block = nn.ModuleList([
                TransformerBlock(
                    embed_dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    sr_ratio=sr_ratios[i],
                    qkv_bias=True,
                )
                for _ in range(depths[i])
            ])
            self.blocks.append(tf_block)
            self.norms.append(nn.LayerNorm(embed_dims[i]))

        self.blocks = nn.ModuleList(self.blocks)
        self.norms = nn.ModuleList(self.norms)

        if self.freeze:
            self.eval()
            self.requires_grad_(False)

    def forward(self, x):
        out = []
        for stage_idx in range(4):
            # embedding
            x, patched_volume_size = self.embeds[stage_idx](x)
            patched_volume_size: list[int]
            B, N, C = x.shape
            
            for blk in self.blocks[stage_idx]:  # type:ignore
                x = blk(x, patched_volume_size)
            x = self.norms[stage_idx](x)
            
            x = x.reshape(B, *patched_volume_size, C).permute(0, 4, 1, 2, 3).contiguous()
            out.append(x)
        return out


class SegFormer3D_Decoder_MM(BaseDecodeHead_3D):
    def __init__(
        self, 
        num_classes:int|None=None, 
        embed_dims:list[int]=[64, 128, 320, 512],
        head_embed_dims:int=256,
        *args, **kwargs
    ):
        if num_classes is None:
            warn("num_classes is not provided, set to head_embed_dims by default.")
            num_classes = head_embed_dims
            
        super().__init__(
            in_channels=embed_dims, 
            channels=head_embed_dims, 
            num_classes=num_classes,
            input_transform="multiple_select",
            in_index=[0, 1, 2, 3],
            *args, **kwargs)
        self.segformer = SegFormerDecoderHead(
            input_feature_dims=embed_dims[::-1],
            decoder_head_embedding_dim=head_embed_dims,
            num_classes=num_classes,
        )

    # SegFormer3D doesn't support torch._dynamo.compile @ 2.5.0.
    # The issue may be caused by inplace interpolate in Decoder forward.
    # HACK Disable compile for now.
    @torch.compiler.disable
    def forward(self, *args, **kwargs):
        num_input_elements = len(args)
        if num_input_elements == 1:
            assert len(args[0]) == 4, f"Invalid number of inputs for SegFormer3D_Decoder_MM: {len(args[0])}"
            segformer_out = self.segformer.forward(*args[0])
        elif num_input_elements == 4:
            segformer_out = self.segformer.forward(*args)
        else:
            raise ValueError(f"Invalid number of inputs for SegFormer3D_Decoder_MM: {num_input_elements}")
        return (segformer_out, )

