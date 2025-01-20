import os
import pdb

from torch import nn
from mmengine.model import BaseModule
from .SegFormer3D import PatchEmbedding, TransformerBlock, cube_root, SegFormerDecoderHead


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
    ):
        super().__init__()

        # 替换为 3D PatchEmbedding (假设此类内部使用 nn.Conv3d)
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

    def forward(self, x):
        out = []
        for stage_idx in range(4):
            # embedding
            x = self.embeds[stage_idx](x)
            B, N, C = x.shape
            for blk in self.blocks[stage_idx]:  # type:ignore
                x = blk(x)
            x = self.norms[stage_idx](x)
            n = cube_root(N)
            
            x = x.reshape(B, n, n, n, C).permute(0, 4, 1, 2, 3).contiguous()
            out.append(x)
        return out


class SegFormer3D_Decoder_MM(SegFormerDecoderHead, BaseModule):
    def __init__(
        self, 
        embed_dims:list[int]=[64, 128, 320, 512],
        head_embed_dims:int=256,
        *args, **kwargs
    ):
        super().__init__(
            input_feature_dims=embed_dims[::-1],
            decoder_head_embedding_dim=head_embed_dims,
            *args, **kwargs
        )


