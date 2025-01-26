import os
import pdb
from typing_extensions import OrderedDict
from abc import abstractmethod

import numpy as np
import torch
from torch import nn, Tensor

from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmengine.utils.misc import is_list_of
from mmengine.structures import BaseDataElement
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup.base import BaseSelfSupervisor



class AutoEncoderSelfSup(BaseSelfSupervisor):
    def __init__(
        self,
        encoder: dict,
        neck: dict | None = None,
        decoder: dict | None = None,
        head: dict | None = None,
        pretrained: str | None = None,
        data_preprocessor: dict | None = None,
        init_cfg: list[dict] | dict | None = None,
        *args,
        **kwargs,
    ) -> None:
        
        encoder_decoder = [MODELS.build(encoder)]
        if neck is not None:
            encoder_decoder.append(MODELS.build(neck))
        if decoder is not None:
            encoder_decoder.append(MODELS.build(decoder))
        encoder_decoder = nn.Sequential(*encoder_decoder)

        super().__init__(
            backbone=encoder_decoder,
            neck=None,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            *args,
            **kwargs,
        )
        self.backbone: BaseModule
        self.neck: BaseModule
        self.head: BaseModule

    @property
    def whole_model_(self) -> nn.Module:
        if self.with_neck:
            return nn.Sequential(self.backbone, self.neck)
        else:
            return self.backbone

    def parse_losses(
        self,
        losses: dict,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        log_vars = []
        for loss_name, loss_value in losses.items():
            if "loss" in loss_name:
                if isinstance(loss_value, Tensor):
                    log_vars.append([loss_name, loss_value.mean()])
                elif is_list_of(loss_value, Tensor):
                    log_vars.append(
                        [loss_name, sum(_loss.mean() for _loss in loss_value)]
                    )
                else:
                    raise TypeError(f"{loss_name} is not a tensor or list of tensors")
            else:
                log_vars.append([loss_name, loss_value])

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)  # type: ignore
        return loss, log_vars  # type: ignore

    @abstractmethod
    def loss(
        self, inputs: list[Tensor], data_samples: list[DataSample], **kwargs
    ) -> dict[str, Tensor]: ...


class VoxelData(BaseDataElement):
    """Data structure for voxel-level annotations or predictions.
    
    All data items in ``data_fields`` of ``VoxelData`` meet the following  
    requirements:
    
    - They all have 4 dimensions in orders of channel, depth, height, and width
    - They should have the same depth, height and width
    
    Examples:
        >>> metainfo = dict(
        ...     vol_id=random.randint(0, 100),
        ...     vol_shape=(random.randint(32,64), 
        ...               random.randint(64,128),
        ...               random.randint(64,128)))
        >>> volume = np.random.randint(0, 255, (4, 32, 64, 64))
        >>> featmap = torch.randint(0, 255, (10, 32, 64, 64))
        >>> voxel_data = VoxelData(metainfo=metainfo,
        ...                        volume=volume, 
        ...                        featmap=featmap)
        >>> print(voxel_data.shape)
        (32, 64, 64)
    """

    @property
    def shape(self):
        """返回体素数据的形状(depth, height, width)。"""
        for v in self.values():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                return v.shape[1:]  # 跳过channel维度
        return None
    
    def __setattr__(self, name, val):
        """设置属性时验证数据维度。"""
        if isinstance(val, (np.ndarray, torch.Tensor)):
            if len(val.shape) != 4:
                raise ValueError(
                    f'The dims of {name} should be 4, but got {len(val.shape)}')
            if hasattr(self, 'shape'):
                if self.shape and self.shape != val.shape[1:]:
                    raise ValueError(
                        f'The shape of {name} does not match the existing shape: '
                        f'expected {self.shape}, but got {val.shape[1:]}')
        super().__setattr__(name, val)
