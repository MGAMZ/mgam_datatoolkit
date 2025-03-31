"""
2024.11.02
Implemented by Yiqin Zhang ~ MGAM.
Used for Rose Thyroid Cell Count project.
"""

from functools import partial

import torch
from torchvision.models import vgg16
from torch import nn

from mmengine.model import BaseModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead, accuracy
from mmseg.utils import SampleList


class DDCB(BaseModule):
    def __init__(self, in_channels, out_channels):
        super(DDCB, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, padding=0, dilation=1),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels + 64, 256, kernel_size=1, padding=0, dilation=1),
            nn.Conv2d(256, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels + 64 + 64, 256, kernel_size=1, padding=0, dilation=1),
            nn.Conv2d(256, 64, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
        )
        self.layer4 = nn.Conv2d(
            in_channels + 64 + 64 + 64,
            out_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
        )

    def forward(self, input):
        x1 = self.layer1(input)
        x2 = torch.cat((input, x1), dim=1)
        x3 = self.layer2(x2)
        x4 = torch.cat((input, x1, x3), dim=1)
        x5 = self.layer3(x4)
        x6 = torch.cat((input, x1, x3, x5), dim=1)
        x7 = self.layer4(x6)
        return x7


class VGG16(BaseModule):
    def __init__(self, torchvision_pretrained: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only use vgg16's feature extraction output,
        # final classification projection will be discard.
        self.base_model = vgg16(torchvision_pretrained).features[:23]  # type:ignore

    def forward(self, input):
        return (self.base_model(input),)


class DSNet(BaseDecodeHead):
    def __init__(self, *args, **kwargs):
        super().__init__(in_channels=512, channels=512, num_classes=1, *args, **kwargs)
        self.ddcb1 = DDCB(512, 512)
        self.ddcb2 = DDCB(512, 512)
        self.ddcb3 = DDCB(512, 512)
        self.layer_last = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1, dilation=1),
            nn.ReLU())
        self.post1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.post2 = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def forward(self, input):
        x1 = input[0]
        x2 = self.ddcb1(x1)
        x3 = self.ddcb2(x1 + x2)
        x4 = self.ddcb3(x1 + x2 + x3)
        x5 = self.layer_last(x1 + x2 + x3 + x4)
        x6 = self.post1(x5)
        x7 = self.post2(x6)
        return x7

    def loss_by_feat(self, seg_logits: torch.Tensor, batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples).squeeze(1)
        loss = dict()

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            loss[loss_decode.loss_name] = loss_decode(
                seg_logits,
                seg_label,
                ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        
        return loss

    def predict_by_feat(self, seg_logits: torch.Tensor, batch_img_metas: list[dict]) -> torch.Tensor:
        return seg_logits