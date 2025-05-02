import pdb
from typing_extensions import Sequence

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mmengine.model import BaseModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ..mm.mmseg_Dev3D import BaseDecodeHead_3D, PixelUnshuffle3D, PixelShuffle3D
from .utils import pad_ensure_conv_out_same_size


class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        stride: int = 1,
        do_res: bool = True,
        norm_type: str = "group",
        n_groups: int|None = None,
        dim: str = "3d",
        grn: bool = False,
    ):
        super().__init__()

        self.do_res = do_res

        assert dim in ["1d", "2d", "3d"]
        self.dim = dim
        if self.dim == "1d":
            conv = nn.Conv1d
        elif self.dim == "2d":
            conv = nn.Conv2d
        elif self.dim == "3d":
            conv = nn.Conv3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad_ensure_conv_out_same_size(kernel_size, stride, 1),
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        elif norm_type == "layer":
            raise NotImplementedError("LayerNorm is deprecated.")

        # Second convolution (Expansion) layer with 1x1x1
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with 1x1x1
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.grn = grn

        if grn:
            if dim == "3d":
                self.grn_beta = nn.Parameter(
                    torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True
                )
                self.grn_gamma = nn.Parameter(
                    torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True
                )
            elif dim == "2d":
                self.grn_beta = nn.Parameter(
                    torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True
                )
                self.grn_gamma = nn.Parameter(
                    torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True
                )
            elif dim == "1d":
                self.grn_beta = nn.Parameter(
                    torch.zeros(1, exp_r * in_channels, 1), requires_grad=True
                )
                self.grn_gamma = nn.Parameter(
                    torch.zeros(1, exp_r * in_channels, 1), requires_grad=True
                )

    def forward(self, x, dummy_tensor=None) -> Tensor:
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input shape based on dim
            if self.dim == "3d":
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == "2d":
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            elif self.dim == "1d":
                gx = torch.norm(x1, p=2, dim=-1, keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(
        self,
        in_channels,
        out_channels,
        exp_r=4,
        kernel_size=7,
        do_res=False,
        norm_type="group",
        dim="3d",
        grn=False,
    ):

        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        if dim == "2d":
            conv = nn.Conv2d
        elif dim == "3d":
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(
        self,
        in_channels,
        out_channels,
        exp_r=4,
        kernel_size=7,
        stride=2,
        do_res=False,
        norm_type="group",
        dim="3d",
        grn=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.resample_do_res = do_res

        self.dim = dim
        if dim == "2d":
            conv = nn.ConvTranspose2d
        elif dim == "3d":
            conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None) -> Tensor:
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape

        if self.dim == "2d":
            x1_padded = torch.nn.functional.interpolate(
                x1, 
                size=[x1.size(-2)+1, x1.size(-1)+1], 
                mode="bilinear", 
                align_corners=False
            )
            # x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        elif self.dim == "3d":
            x1_padded = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))
        else:
            raise ValueError("Invalid dimension")

        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == "2d":
                res_padded = torch.nn.functional.interpolate(
                    res, size=[res.size(-2)+1, res.size(-1)+1], mode="bilinear", align_corners=False
                )
                
                # res = torch.nn.functional.pad(res, (1, 0, 1, 0))
            elif self.dim == "3d":
                res_padded = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            
            x1_padded = x1_padded + res_padded

        return x1_padded


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        if dim == "2d":
            conv = nn.ConvTranspose2d
        elif dim == "3d":
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class MedNeXt(nn.Module):

    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r: int | list[int] = 4,  # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,  # Ofcourse can test kernel_size
        enc_kernel_size: int | None = None,
        dec_kernel_size: int | None = None,
        deep_supervision: bool = False,  # Can be used to test deep supervision
        do_res: bool = False,  # Can be used to individually test residual connection
        do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
        checkpoint_style: bool | None = None,  # Either inside block or outside block
        block_counts: list = [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ],  # Can be used to test staging ratio:
        # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type="group",
        dim="3d",  # 2d or 3d
        grn=False,
    ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, "outside_block"]
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == "outside_block":
            self.outside_block_checkpointing = True
        assert dim in ["2d", "3d"]

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == "2d":
            conv = nn.Conv2d
        elif dim == "3d":
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    exp_r=exp_r[0],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 2,
                    out_channels=n_channels * 2,
                    exp_r=exp_r[1],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 4,
                    out_channels=n_channels * 4,
                    exp_r=exp_r[2],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 8,
                    out_channels=n_channels * 8,
                    exp_r=exp_r[3],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 16,
                    out_channels=n_channels * 16,
                    exp_r=exp_r[4],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[4])
            ]
        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 8,
                    out_channels=n_channels * 8,
                    exp_r=exp_r[5],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[5])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 4,
                    out_channels=n_channels * 4,
                    exp_r=exp_r[6],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[6])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 2,
                    out_channels=n_channels * 2,
                    exp_r=exp_r[7],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[7])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    exp_r=exp_r[8],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[8])
            ]
        )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(Tensor([1.0]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(
                in_channels=n_channels * 2, n_classes=n_classes, dim=dim
            )
            self.out_2 = OutBlock(
                in_channels=n_channels * 4, n_classes=n_classes, dim=dim
            )
            self.out_3 = OutBlock(
                in_channels=n_channels * 8, n_classes=n_classes, dim=dim
            )
            self.out_4 = OutBlock(
                in_channels=n_channels * 16, n_classes=n_classes, dim=dim
            )

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):
        
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)

            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


class MM_MedNext_Encoder(BaseModule):
    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        exp_r=4,  # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,  # Ofcourse can test kernel_size
        enc_kernel_size: int | None = None,
        dec_kernel_size: int | None = None,
        deep_supervision: bool = False,  # Can be used to test deep supervision
        do_res: bool = False,  # Can be used to individually test residual connection
        do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
        use_checkpoint: bool = False,  # Either inside block or outside block
        block_counts: Sequence = [2,2,2,2,2,2,2,2,2],  # Can be used to test staging ratio:
        # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type="group",
        dim="2d",  # 2d or 3d
        grn=False,
        freeze:bool=False,
        pixel_unshuffle:Sequence[int]|int|None=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.do_ds = deep_supervision
        self.freeze = freeze
        assert dim in ["2d", "3d"]
        self.use_checkpoint = use_checkpoint
        
        if dim == "2d":
            conv = nn.Conv2d
        elif dim == "3d":
            conv = nn.Conv3d
        
        # NOTE The stem uses the actual embedding dims.
        #      The pixel unshuffle will increase the number of channels,
        #      and is after the stem forward,
        #      so mednext layers' embed dims are configured to a larger channel.
        self.stem = conv(in_channels, embed_dims, kernel_size=1)

        if type(exp_r) == int:
            exp_r = [exp_r] * len(block_counts)
        else:
            assert isinstance(exp_r, list)
            
        # When 3D Model, should specify the pixel_unshuffle ratio for each dimension;
        # when 2D Model, only support the same ratio on X and Y.
        if pixel_unshuffle is not None:
            if dim == "2d":
                assert isinstance(pixel_unshuffle, int)
                self.pixel_unshuffle = nn.PixelUnshuffle(pixel_unshuffle)
            elif dim == "3d":
                self.pixel_unshuffle = PixelUnshuffle3D(pixel_unshuffle)
            
            embed_dims *= int(np.prod(pixel_unshuffle))

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        self.enc_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    exp_r=exp_r[0],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[0])
            ]
        )

        self.down_0 = MedNeXtDownBlock(
            in_channels=embed_dims,
            out_channels=2 * embed_dims,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=embed_dims * 2,
                    out_channels=embed_dims * 2,
                    exp_r=exp_r[1],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[1])
            ]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * embed_dims,
            out_channels=4 * embed_dims,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=embed_dims * 4,
                    out_channels=embed_dims * 4,
                    exp_r=exp_r[2],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[2])
            ]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * embed_dims,
            out_channels=8 * embed_dims,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=embed_dims * 8,
                    out_channels=embed_dims * 8,
                    exp_r=exp_r[3],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[3])
            ]
        )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * embed_dims,
            out_channels=16 * embed_dims,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=embed_dims * 16,
                    out_channels=embed_dims * 16,
                    exp_r=exp_r[4],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[4])
            ]
        )

        if self.freeze:
            self.eval()
            self.requires_grad_(False)

        # HACK Use for grad visualization
        # self.register_backward_hook(grad_hist_and_pixelwise_vis_hook)
        # self.register_backward_hook(log_grad)

    def forward(self, x: Tensor):
        # [B, 1, Z(Opt.), Y, X] -> [B, C, Z(Opt.), Y, X]
        x = self.stem(x)
        if hasattr(self, "pixel_unshuffle"):
            x = self.pixel_unshuffle(x)
        
        if self.use_checkpoint:
            x_res_0 = checkpoint(self.enc_block_0, x, use_reentrant=False)
            x = checkpoint(self.down_0, x_res_0, use_reentrant=False)
            x_res_1 = checkpoint(self.enc_block_1, x, use_reentrant=False)
            x = checkpoint(self.down_1, x_res_1, use_reentrant=False)
            x_res_2 = checkpoint(self.enc_block_2, x, use_reentrant=False)
            x = checkpoint(self.down_2, x_res_2, use_reentrant=False)
            x_res_3 = checkpoint(self.enc_block_3, x, use_reentrant=False)
            x = checkpoint(self.down_3, x_res_3, use_reentrant=False)
            x = checkpoint(self.bottleneck, x, use_reentrant=False)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)
            x = self.bottleneck(x)
        
        # HACK Use for activation map visualization
        # vis_act_map(x_res_0, x_res_1, x_res_2, x_res_3, x)
        # vis_tSNE(x_res_0, x_res_1, x_res_2, x_res_3, x)
        # vis_PCA(x_res_0, x_res_1, x_res_2, x_res_3, x)
        # log_act(x_res_0, x_res_1, x_res_2, x_res_3, x)
        
        return (x_res_0, x_res_1, x_res_2, x_res_3, x)

# HACK Use for grad visualization
def grad_hist_and_pixelwise_vis_hook(module:nn.Module, grad_input:Tensor, grad_output:Tensor):
    """可视化MM_MedNext_Encoder的输入和输出梯度
    
    Args:
        module: 调用该hook的模块
        grad_input: 输入的梯度（tuple）
        grad_output: 输出的梯度（tuple）
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from datetime import datetime
    from matplotlib.font_manager import FontProperties
    
    # 创建保存路径
    save_dir = os.path.join("visualization")
    os.makedirs(save_dir, exist_ok=True)
    font = FontProperties(fname="/mnt/c/Windows/Fonts/simhei.ttf", size=14)
    cmap = "GnBu"
    alpha = 0.8

    grad_in = grad_input[0].mean(dim=(0,1)).detach().cpu().numpy()
    grad_out = grad_output[0].mean(dim=(0,1)).detach().cpu().numpy()
    
    # 计算统一的最大最小值用于归一化
    all_grads = np.concatenate([grad_in.flatten(), grad_out.flatten()])
    vmin = np.percentile(all_grads, 65)
    vmax = np.percentile(all_grads, 95)
    
    # 创建图形和网格
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.05, 1, 0.15])
    
    # 第一行：颜色条
    cbar_ax = fig.add_subplot(gs[0, :])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
        cax=cbar_ax, 
        orientation='horizontal',
        alpha=alpha)
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')
    
    # 第二行：热图
    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(grad_in, 
                   cmap=cmap, 
                   alpha=alpha,
                   norm=norm)
    ax1.set_title("输入梯度", fontsize=14, fontproperties=font)
    ax1.text(0, 20, 
             f"Std: {grad_in.std():.5f}", 
             fontsize=12, 
             color='white')
    
    ax2 = fig.add_subplot(gs[1, 1])
    im2 = ax2.imshow(grad_out, 
                   cmap=cmap, 
                   alpha=alpha,
                   norm=norm)
    ax2.set_title("输出梯度", fontsize=14, fontproperties=font)
    ax2.text(0, 20, 
             f"Std: {grad_out.std():.5f}", 
             fontsize=12, 
             color='white', 
             fontproperties=font)
    
    # 第三行：直方图（对数尺度）
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(grad_in.flatten(), bins=50, alpha=alpha, color='skyblue')
    ax3.set_ylabel("频率", fontproperties=font)
    ax3.set_yscale('log')
    
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(grad_out.flatten(), bins=50, alpha=alpha, color='skyblue')
    ax4.set_yscale('log')
    
    # 设置x轴范围一致
    x_min = min(grad_in.min(), grad_out.min())
    x_max = max(grad_in.max(), grad_out.max())
    ax3.set_xlim(x_min, x_max)
    ax4.set_xlim(x_min, x_max)
    
    # 保存图像
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"grad_pixel_hist_vis_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存梯度可视化: {save_path}")
    plt.close()

# HACK Use for activation map visualization
def vis_act_map(x_res_0:Tensor, x_res_1:Tensor, x_res_2:Tensor,
                x_res_3:Tensor, x:Tensor,
                cmap="GnBu", alpha=0.8):
    x_res_0 = x_res_0.mean(dim=(0,1)).detach().cpu().numpy()
    x_res_1 = x_res_1.mean(dim=(0,1)).detach().cpu().numpy()
    x_res_2 = x_res_2.mean(dim=(0,1)).detach().cpu().numpy()
    x_res_3 = x_res_3.mean(dim=(0,1)).detach().cpu().numpy()
    x = x.mean(dim=(0,1)).detach().cpu().numpy()
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1,5, figsize=(15,5))
    norm = Normalize(vmin=-0.03, vmax=0.05)
    norm = Normalize()
    for i, arr in enumerate([x_res_0, x_res_1, x_res_2, x_res_3, x]):
        axes[i].set_title(f"Layer {i}")
        axes[i].imshow(arr, norm=norm, cmap=cmap, alpha=alpha)
        print(f"Layer {i} std: {arr.var()}")
    
    cbar_ax = fig.add_axes(rect=(0.15, 0.05, 0.7, 0.03))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cbar_ax, orientation="horizontal", alpha=alpha)
    
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.01, hspace=0)
    fig.savefig("visualization/ActivationMap.png", dpi=300)

# HACK Use for t-SNE analysis
def vis_tSNE(x_res_0:Tensor, x_res_1:Tensor, x_res_2:Tensor, x_res_3:Tensor, x:Tensor):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    
    x_res_0 = x_res_0.detach().mean(dim=(2,3)).cpu().numpy()
    x_res_1 = x_res_1.detach().mean(dim=(2,3)).cpu().numpy()
    x_res_2 = x_res_2.detach().mean(dim=(2,3)).cpu().numpy()
    x_res_3 = x_res_3.detach().mean(dim=(2,3)).cpu().numpy()
    x = x.detach().mean(dim=(2,3)).cpu().numpy()
    
    tsne = TSNE(n_components=2, random_state=0)
    x_res_0 = tsne.fit(x_res_0)
    x_res_1 = tsne.fit(x_res_1)
    x_res_2 = tsne.fit(x_res_2)
    x_res_3 = tsne.fit(x_res_3)
    x = tsne.fit(x)
    
    fig, axes = plt.subplots(1,5, figsize=(15,5))
    for i, arr in enumerate([x_res_0, x_res_1, x_res_2, x_res_3, x]):
        axes[i].set_title(f"Layer {i}")
        axes[i].scatter(arr[:, 0], arr[:, 1], alpha=0.9)
    
    fig.tight_layout()
    fig.savefig("visualization/tSNE.png", dpi=300)

# HACK Use for PCA
def vis_PCA(x_res_0:Tensor, x_res_1:Tensor, x_res_2:Tensor, x_res_3:Tensor, x:Tensor):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    x_res_0 = x_res_0.detach().mean(dim=(2,3)).cpu().numpy()
    x_res_1 = x_res_1.detach().mean(dim=(2,3)).cpu().numpy()
    x_res_2 = x_res_2.detach().mean(dim=(2,3)).cpu().numpy()
    x_res_3 = x_res_3.detach().mean(dim=(2,3)).cpu().numpy()
    x = x.detach().mean(dim=(2,3)).cpu().numpy()
    
    pca = PCA(n_components=2)
    x_res_0 = pca.fit_transform(x_res_0)
    x_res_1 = pca.fit_transform(x_res_1)
    x_res_2 = pca.fit_transform(x_res_2)
    x_res_3 = pca.fit_transform(x_res_3)
    x = pca.fit_transform(x)
    
    fig, axes = plt.subplots(1,5, figsize=(15,5))
    for i, arr in enumerate([x_res_0, x_res_1, x_res_2, x_res_3, x]):
        axes[i].set_title(f"Layer {i}")
        axes[i].scatter(arr[:, 0], arr[:, 1], alpha=0.9)
    
    fig.tight_layout()
    fig.savefig("visualization/PCA.png", dpi=300)

# HACK
def log_grad(module:nn.Module, grad_input:Tensor, grad_output:Tensor):
    import os
    import numpy as np
    from time import time
    save_dir = "visualization/grad"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"grad_{time()}.npz")
    if grad_input[0] is not None and grad_output[0] is not None:
        np.savez_compressed(
            save_path, 
            grad_in=grad_input[0].detach().cpu().numpy(), 
            grad_out=grad_output[0].detach().cpu().numpy()
        )

# HACK
def log_act(x_res_0:Tensor, x_res_1:Tensor, x_res_2:Tensor, x_res_3:Tensor, x:Tensor):
    import os
    import numpy as np
    from time import time
    save_dir = "visualization/act"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"act_{time()}.npz")
    np.savez_compressed(
        save_path, 
        layer0=x_res_0.detach().cpu().numpy(),
        layer1=x_res_1.detach().cpu().numpy(),
        layer2=x_res_2.detach().cpu().numpy(),
        layer3=x_res_3.detach().cpu().numpy(),
        layer4=x.detach().cpu().numpy()
    )


class MM_MedNext_Decoder(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        num_classes: int,
        exp_r=4,  # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,  # Ofcourse can test kernel_size
        do_res: bool = False,  # Can be used to individually test residual connection
        do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
        use_checkpoint: bool = False,
        block_counts: list = [2,2,2,2,2,2,2,2,2],  # Can be used to test staging ratio:
        # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        deep_supervision: bool = False,
        norm_type = "group",
        dim = "2d",  # 2d or 3d
        grn = False,
        pixel_shuffle: Sequence[int]|int|None = None,
        *args,
        **kwargs,
    ):
        assert dim in ["2d", "3d"]
        super().__init__(
            *args,
            **kwargs,
        )
        self.deep_supervision = deep_supervision
        if type(exp_r) == int:
            exp_r = [exp_r] * len(block_counts)
        else:
            assert isinstance(exp_r, list)

        if use_checkpoint:
            self.checkpoint = lambda f, x: checkpoint(f, x, use_reentrant=False)
        else:
            self.checkpoint = lambda f, x: f(x)

        # NOTE Out projection is not influenced by pixel shuffle operation.
        #      The tensor will have already been shuffled back to the actual size of input,
        #      so the out projection will receive the actual embed_dims.
        self.out_0 = OutBlock(in_channels=embed_dims, n_classes=num_classes, dim=dim)
        
        # When 3D Model, should specify the pixel_shuffle ratio for each dimension;
        # when 2D Model, only support the same ratio on X and Y.
        if pixel_shuffle is not None:
            if dim == "2d":
                assert isinstance(pixel_shuffle, int)
                self.pixel_shuffle = nn.PixelShuffle(pixel_shuffle)
            elif dim == "3d":
                self.pixel_shuffle = PixelShuffle3D(pixel_shuffle)
            
            embed_dims *= int(np.prod(pixel_shuffle))

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * embed_dims,
            out_channels=8 * embed_dims,
            exp_r=exp_r[5],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=embed_dims * 8,
                    out_channels=embed_dims * 8,
                    exp_r=exp_r[5],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[5])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * embed_dims,
            out_channels=4 * embed_dims,
            exp_r=exp_r[6],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=embed_dims * 4,
                    out_channels=embed_dims * 4,
                    exp_r=exp_r[6],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[6])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * embed_dims,
            out_channels=2 * embed_dims,
            exp_r=exp_r[7],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=embed_dims * 2,
                    out_channels=embed_dims * 2,
                    exp_r=exp_r[7],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[7])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * embed_dims,
            out_channels=embed_dims,
            exp_r=exp_r[8],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    exp_r=exp_r[8],
                    kernel_size=kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for _ in range(block_counts[8])
            ]
        )

        self.block_counts = block_counts

        if deep_supervision:
            self.out_1 = OutBlock(
                in_channels=embed_dims * 2, n_classes=num_classes, dim=dim
            )
            self.out_2 = OutBlock(
                in_channels=embed_dims * 4, n_classes=num_classes, dim=dim
            )
            self.out_3 = OutBlock(
                in_channels=embed_dims * 8, n_classes=num_classes, dim=dim
            )
            self.out_4 = OutBlock(
                in_channels=embed_dims * 16, n_classes=num_classes, dim=dim
            )

    def forward(self, inputs):
        (x_res_0, x_res_1, x_res_2, x_res_3, x) = inputs
        if self.deep_supervision:
            x_ds_4 = self.checkpoint(self.out_4, x)

        x_up_3 = self.checkpoint(self.up_3, x)
        dec_x = x_res_3 + x_up_3
        x = self.checkpoint(self.dec_block_3, dec_x)
        if self.deep_supervision:
            x_ds_3 = self.checkpoint(self.out_3, x)
        del x_res_3, x_up_3

        x_up_2 = self.checkpoint(self.up_2, x)
        dec_x = x_res_2 + x_up_2
        x = self.checkpoint(self.dec_block_2, dec_x)
        if self.deep_supervision:
            x_ds_2 = self.checkpoint(self.out_2, x)
        del x_res_2, x_up_2

        x_up_1 = self.checkpoint(self.up_1, x)
        dec_x = x_res_1 + x_up_1
        x = self.checkpoint(self.dec_block_1, dec_x)
        if self.deep_supervision:
            x_ds_1 = self.checkpoint(self.out_1, x)
        del x_res_1, x_up_1

        x_up_0 = self.checkpoint(self.up_0, x)
        dec_x = x_res_0 + x_up_0
        x = self.checkpoint(self.dec_block_0, dec_x)
        del x_res_0, x_up_0, dec_x

        # out projection
        if hasattr(self, "pixel_shuffle"):
            x = self.pixel_shuffle(x)
        x = self.checkpoint(self.out_0, x)
        
        if self.deep_supervision:
            # deep_out element: Tensor[N, C, Z, Y, X]
            return (x, x_ds_1, x_ds_2, x_ds_3, x_ds_4)
        else:
            return (x,)


class MM_MedNext_Decoder_2D(BaseDecodeHead):
    def __init__(
        self,
        embed_dims: int,
        num_classes: int,
        exp_r=4,
        kernel_size: int = 7,
        block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],
        deep_supervision: bool = False,
        use_checkpoint: bool = False,
        norm_type="group",
        grn=False,
        freeze:bool = False,
        pixel_shuffle: int|None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels=[
                embed_dims,
                embed_dims * 2,
                embed_dims * 4,
                embed_dims * 8,
                embed_dims * 16,
            ],
            channels=embed_dims,
            num_classes=num_classes,
            input_transform="multiple_select",
            in_index=[0, 1, 2, 3, 4],
            *args,
            **kwargs)
        
        self.freeze = freeze
        self.mednext = MM_MedNext_Decoder(
            embed_dims=embed_dims,
            num_classes=num_classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            block_counts=block_counts,
            deep_supervision=deep_supervision,
            use_checkpoint=use_checkpoint,
            norm_type=norm_type,
            dim="2d",
            grn=grn,
            pixel_shuffle=pixel_shuffle)

        if self.freeze:
            self.eval()
            self.requires_grad_(False)

    def forward(self, inputs):
        return self.mednext(inputs)[0]


class MM_MedNext_Decoder_3D(BaseDecodeHead_3D):
    def __init__(
        self,
        embed_dims: int,
        num_classes: int,
        exp_r=4,
        kernel_size: int = 7,
        block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],
        deep_supervision: bool = False,
        use_checkpoint: bool = False,
        norm_type="group",
        grn=False,
        freeze:bool=False,
        pixel_shuffle=None,
        avgpool_XY:bool=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels=[
                embed_dims,
                embed_dims * 2,
                embed_dims * 4,
                embed_dims * 8,
                embed_dims * 16,
            ],
            channels=embed_dims,
            num_classes=num_classes,
            input_transform="multiple_select",
            in_index=[0, 1, 2, 3, 4],
            *args,
            **kwargs)
        
        self.avgpool_XY = avgpool_XY
        self.freeze = freeze
        self.mednext = MM_MedNext_Decoder(
            embed_dims=embed_dims,
            num_classes=num_classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            block_counts=block_counts,
            deep_supervision=deep_supervision,
            use_checkpoint=use_checkpoint,
            norm_type=norm_type,
            grn=grn,
            dim="3d",
            pixel_shuffle=pixel_shuffle)

        if self.freeze:
            self.eval()
            self.requires_grad_(False)

    def forward(self, inputs):
        dec_out = self.mednext(inputs)
        if self.avgpool_XY:
            return [x.mean(dim=(-1,-2)) for x in dec_out]
        else:
            return dec_out

# NOTE This class is decrecated and is only used for 
# NOTE implementations of Sarcopenia project, i.e., weight loading.
class MM_MedNext_Decoder_Vallina(BaseDecodeHead):
    def __init__(self,
        embed_dims: int,
        num_classes: int, 
        out_channels: int, 
        exp_r = 4,                            # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                      # Ofcourse can test kernel_size
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio: 
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type = 'group',
        dim = '2d',                                # 2d or 3d
        grn = False,
        **kwargs,
    ):
        super().__init__(in_channels=[embed_dims,
                                      embed_dims*2,
                                      embed_dims*4,
                                      embed_dims*8,
                                      embed_dims*16],
                         channels=embed_dims,
                         num_classes=num_classes,
                         out_channels=out_channels,
                         input_transform='multiple_select',
                         in_index=[0,1,2,3,4],
                         **kwargs)
        
        if type(exp_r) == int:
            exp_r = [exp_r] * len(block_counts)
        else:
            assert isinstance(exp_r, list)
        
        self.up_3 = MedNeXtUpBlock(
            in_channels=16*embed_dims,
            out_channels=8*embed_dims,
            exp_r=exp_r[5],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=embed_dims*8,
                out_channels=embed_dims*8,
                exp_r=exp_r[5],
                kernel_size=kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for _ in range(block_counts[5])]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8*embed_dims,
            out_channels=4*embed_dims,
            exp_r=exp_r[6],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=embed_dims*4,
                out_channels=embed_dims*4,
                exp_r=exp_r[6],
                kernel_size=kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for _ in range(block_counts[6])]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4*embed_dims,
            out_channels=2*embed_dims,
            exp_r=exp_r[7],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=embed_dims*2,
                out_channels=embed_dims*2,
                exp_r=exp_r[7],
                kernel_size=kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for _ in range(block_counts[7])]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2*embed_dims,
            out_channels=embed_dims,
            exp_r=exp_r[8],
            kernel_size=kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=embed_dims,
                out_channels=embed_dims,
                exp_r=exp_r[8],
                kernel_size=kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for _ in range(block_counts[8])]
        )

        self.block_counts = block_counts
    
    def forward(self, inputs):
        (x_res_0, x_res_1, x_res_2, x_res_3, x) = inputs
        
        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3 
        x = self.dec_block_3(dec_x)

        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2 
        x = self.dec_block_2(dec_x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1 
        x = self.dec_block_1(dec_x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0 
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.cls_seg(x)
        return x


if __name__ == "__main__":
    toy_model = MedNeXt(
        in_channels=1,
        n_channels=8,
        n_classes=4,
        dim='3d')
    toy_input = torch.randn(1, 1, 64, 64, 64)
    toy_output = toy_model(toy_input)
    print(f"Input shape: {toy_input.shape}")