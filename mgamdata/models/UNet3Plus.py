import pdb
from typing_extensions import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# 根据维度选择相应的组件
def get_components(dim):
    """根据维度返回相应的卷积、批归一化和池化层"""
    if dim == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d
    elif dim == 3:
        return nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d
    else:
        raise ValueError(f"维度必须是2或3，当前值: {dim}")


def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding='same',
               is_bn=True, is_relu=True, n=2, dim=2):
    """ 支持2D/3D的卷积块 """
    Conv, BatchNorm, _ = get_components(dim)
    
    # 根据维度调整kernel_size和stride为元组形式
    if isinstance(kernel_size, int):
        kernel_size = tuple([kernel_size] * dim)
    if isinstance(stride, int):
        stride = tuple([stride] * dim)
    
    layers = []
    for i in range(1, n + 1):
        conv = Conv(in_channels=in_channels if i == 1 else out_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding if padding != 'same' else 'same',
                    bias=not is_bn)
        layers.append(conv)
        
        if is_bn:
            layers.append(BatchNorm(out_channels))
        
        if is_relu:
            layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)


def dot_product(seg, cls):
    if seg.dim() == 4:  # 2D case
        b, n, h, w = seg.shape
        seg = seg.view(b, n, -1)
    else:  # 3D case
        b, n, h, w, d = seg.shape
        seg = seg.view(b, n, -1)
    
    cls = cls.unsqueeze(-1)  # Add an extra dimension for broadcasting
    final = torch.einsum("bik,bi->bik", seg, cls)
    
    if seg.dim() == 4:  # 2D case
        final = final.view(b, n, h, w)
    else:  # 3D case
        final = final.view(b, n, h, w, d)
    
    return final


class UNet3Plus(nn.Module):
    def __init__(self,
                 input_shape:Sequence[int],
                 output_channels:int,
                 filters:Sequence[int]=[64, 128, 256, 512, 1024],
                 deep_supervision:bool=False,
                 ClassificationGuidedModule:bool=False,
                 dim:int=2,
                 use_torch_checkpoint:bool=False):
        """
        UNet3Plus model with optional deep supervision and classification guided module.
        
        Args:
            input_shape (list): Input shape of the model [channels, height, width] or [channels, height, width, depth].
            output_channels (int): Number of output channels.
            deep_supervision (bool): Whether to use deep supervision.
            ClassificationGuidedModule (bool): Whether to use classification guided module.
            dim (int): Dimensionality of data (2 for 2D, 3 for 3D).
        """
        
        super(UNet3Plus, self).__init__()
        self.dim = dim
        self.deep_supervision = deep_supervision
        self.CGM = deep_supervision and ClassificationGuidedModule
        self.use_torch_checkpoint = use_torch_checkpoint
        
        Conv, BatchNorm, MaxPool = get_components(dim)

        self.filters = filters
        self.cat_channels = self.filters[0]
        self.cat_blocks = len(self.filters)
        self.upsample_channels = self.cat_blocks * self.cat_channels

        # Encoder
        self.e1 = ConvBlock(input_shape[0], self.filters[0], dim=dim)
        self.e2 = nn.Sequential(
            MaxPool(2),
            ConvBlock(self.filters[0], self.filters[1], dim=dim)
        )
        self.e3 = nn.Sequential(
            MaxPool(2),
            ConvBlock(self.filters[1], self.filters[2], dim=dim)
        )
        self.e4 = nn.Sequential(
            MaxPool(2),
            ConvBlock(self.filters[2], self.filters[3], dim=dim)
        )
        self.e5 = nn.Sequential(
            MaxPool(2),
            ConvBlock(self.filters[3], self.filters[4], dim=dim)
        )

        # Classification Guided Module
        if self.CGM:
            self.cgm = nn.Sequential(
                nn.Dropout(0.5),
                Conv(self.filters[4], 2, kernel_size=1, padding=0),
                nn.AdaptiveMaxPool3d(1) if dim == 3 else nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                nn.Sigmoid()
            )
        else:
            self.cgm = None

        # Decoder
        self.d4 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[1], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[2], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[3], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[4], self.cat_channels, n=1, dim=dim)
        ])
        self.d4_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1, dim=dim)

        self.d3 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[1], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[2], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[4], self.cat_channels, n=1, dim=dim)
        ])
        self.d3_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1, dim=dim)

        self.d2 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[1], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1, dim=dim),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[4], self.cat_channels, n=1, dim=dim)
        ])
        self.d2_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1, dim=dim)

        self.d1 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1, dim=dim),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1, dim=dim),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1, dim=dim),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1, dim=dim),
            ConvBlock(self.filters[4], self.cat_channels, n=1, dim=dim)
        ])
        self.d1_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1, dim=dim)

        self.final = Conv(self.upsample_channels, output_channels, kernel_size=1)

        # Deep Supervision
        if self.deep_supervision:
            self.deep_sup = nn.ModuleList([
                ConvBlock(self.upsample_channels, output_channels, n=1, is_bn=False, is_relu=False, dim=dim)
                for _ in range(3)
            ] + [ConvBlock(self.filters[4], output_channels, n=1, is_bn=False, is_relu=False, dim=dim)]
            )
        else:
            self.deep_sup = None

    def _maybe_checkpoint(self, func, *args):
        """使用torch的checkpoint机制来节省内存"""
        if self.use_torch_checkpoint:
            return checkpoint(func, *args)
        else:
            return func(*args)

    def forward(self, x) -> torch.Tensor:
        # Encoder
        e1 = self.e1(x)
        e2 = self._maybe_checkpoint(self.e2, e1)
        e3 = self._maybe_checkpoint(self.e3, e2)
        e4 = self._maybe_checkpoint(self.e4, e3)
        e5 = self._maybe_checkpoint(self.e5, e4)

        # Classification Guided Module
        if self.CGM:
            assert self.cgm is not None, "Classification Guided Module is enabled but CGM layer is not defined."
            cls = self.cgm(e5)
            cls = torch.argmax(cls, dim=1).float()

        # 根据维度选择适当的池化和上采样操作
        if self.dim == 2:
            max_pool = F.max_pool2d
            upsample = lambda x, scale: F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
        else:  # dim == 3
            max_pool = F.max_pool3d
            upsample = lambda x, scale: F.interpolate(x, scale_factor=scale, mode='trilinear', align_corners=True)

        # Decoder
        d4 = [
            max_pool(e1, 8),
            max_pool(e2, 4),
            max_pool(e3, 2),
            e4,
            upsample(e5, 2)
        ]
        d4 = [conv(d) for conv, d in zip(self.d4, d4)]
        d4 = torch.cat(d4, dim=1)
        d4 = self._maybe_checkpoint(self.d4_conv, d4)

        d3 = [
            max_pool(e1, 4),
            max_pool(e2, 2),
            e3,
            upsample(d4, 2),
            upsample(e5, 4)
        ]
        d3 = [conv(d) for conv, d in zip(self.d3, d3)]
        d3 = torch.cat(d3, dim=1)
        d3 = self._maybe_checkpoint(self.d3_conv, d3)

        d2 = [
            max_pool(e1, 2),
            e2,
            upsample(d3, 2),
            upsample(d4, 4),
            upsample(e5, 8)
        ]
        d2 = [conv(d) for conv, d in zip(self.d2, d2)]
        d2 = torch.cat(d2, dim=1)
        d2 = self._maybe_checkpoint(self.d2_conv, d2)

        d1 = [
            e1,
            upsample(d2, 2),
            upsample(d3, 4),
            upsample(d4, 8),
            upsample(e5, 16)
        ]
        d1 = [conv(d) for conv, d in zip(self.d1, d1)]
        d1 = torch.cat(d1, dim=1)
        d1 = self.d1_conv(d1)
        d1 = self._maybe_checkpoint(self.final, d1)

        outputs = [d1]

        # Deep Supervision
        if self.deep_supervision:
            assert self.deep_sup is not None, "Deep supervision is enabled but deep supervision layers are not defined."
            outputs.extend([
                upsample(self.deep_sup[0](d2), 2),
                upsample(self.deep_sup[1](d3), 4),
                upsample(self.deep_sup[2](d4), 8),
                upsample(self.deep_sup[3](e5), 16)
            ])

        # Classification Guided Module
        if self.CGM:
            outputs = [dot_product(out, cls) for out in outputs]

        if self.deep_supervision:
            return [F.sigmoid(out) for out in outputs]
        else:
            return F.sigmoid(outputs[0])



if __name__ == "__main__":
    # 2D 示例
    unet_3P_2d = UNet3Plus(input_shape=[1, 320, 320],
                          output_channels=4,
                          dim=2)
    x_2d = torch.randn(3, 1, 320, 320)
    output_2d = unet_3P_2d(x_2d)
    print(f"2D 输出形状: {output_2d.shape}")
    
    # 3D 示例
    unet_3P_3d = UNet3Plus(input_shape=[1, 64, 64, 64],
                           output_channels=4,
                           dim=3)
    x_3d = torch.randn(3, 1, 64, 64, 64)
    output_3d = unet_3P_3d(x_3d)
    print(f"3D 输出形状: {output_3d.shape}")