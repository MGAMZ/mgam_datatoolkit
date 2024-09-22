import os
import os.path as osp
import pdb
from pprint import pprint
from tqdm import tqdm
from typing import Any

import torch
import numpy as np
from torch import Tensor

from mmengine.structures import BaseDataElement
from mmdet.models.losses import CrossEntropyLoss, DiceLoss
from mmdet.models.losses.dice_loss import dice_loss as vanilla_dice




class CrossEntropyLoss_AlignChannel(CrossEntropyLoss):
    def forward(self, 
                cls_score:Tensor, 
                label:Tensor|BaseDataElement, 
                **kwargs):
        if not isinstance(label, Tensor):
            label = label.gt_seg_map.index

        if cls_score.ndim == label.ndim:
            label = label.argmax(dim=1)
        return super().forward(cls_score, label, **kwargs)


# Remastered Dice with Mask on empty background.
class DiceLoss_Remastered(DiceLoss):
    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None,
                ignore_index=None):
        if isinstance(target, BaseDataElement):
            target = target.gt_seg_map.channel
        return super().forward(
            pred, target, weight, reduction_override, avg_factor, ignore_index)



class Axial_L3_Loss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bce = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, cls_score:Tensor, label:BaseDataElement|Tensor):
        '''
            cls_score: [B, D]
            label:
                - gt_L3_location: [B, D]
        '''
        if isinstance(label, BaseDataElement):
            label = label.gt_L3_location
        return self.bce(cls_score, label)




if __name__ == '__main__':
    from aitrox.utils.structure import VolumeData
    
    loss = DiceLoss_AxialMask(21)
    
    pixel = VolumeData(index=Tensor([*[0]*20, *[1]*21, *[0]*20])[None, :, None, None])
    target = BaseDataElement(gt_seg_map=pixel)
    
    mask = loss._genrate_mask(target)
    print(mask.squeeze())
    pdb.set_trace()
