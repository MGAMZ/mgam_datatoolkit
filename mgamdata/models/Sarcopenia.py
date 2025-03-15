import pdb

import numpy as np
import torch
from torch import Tensor

from mmengine.model import BaseModule
from ..mm.mmseg_Dev3D import BaseDecodeHead_3D, Seg3DDataSample


class L3LocationDecoder(BaseDecodeHead_3D):
    def __init__(self, 
                 embed_dims:list[int], 
                 Z_lengths:list[int], 
                 threshold:float=0.3, 
                 hidden_feat_HW:int=32,
                 loss_weight:float=1., 
                 use_checkpoint:bool=False,
                 *args, **kwargs):
        assert len(embed_dims) == len(Z_lengths)
        super().__init__(in_channels=embed_dims,
                         channels=1,
                         num_classes=1,
                         in_index=list(range(len(embed_dims))),
                         input_transform='multiple_select',
                         threshold=threshold,
                         *args, **kwargs
        )
        self.embed_dims = embed_dims
        self.Z_lengths = Z_lengths
        self.hidden_feat_HW = hidden_feat_HW
        self.threshold = threshold
        self.loss_weight = loss_weight
        self.use_checkpoint = use_checkpoint
        in_feat_C = sum(embed_dims)
        hidden_C = embed_dims[0]

        # extractor
        self.z_extractor = torch.nn.ModuleList([
            torch.nn.Conv3d(in_channels=in_feat_C, out_channels=hidden_C, kernel_size=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(in_channels=hidden_C, out_channels=hidden_C, kernel_size=1),
            torch.nn.LeakyReLU()])
        for _ in range(8):
            self.z_extractor.extend([
                torch.nn.Conv3d(in_channels=hidden_C, out_channels=hidden_C, kernel_size=5, padding=2),
                torch.nn.LeakyReLU()])
        
        # downsample
        self.downsample = torch.nn.ModuleList()
        for _ in range(4):
            self.downsample.extend([
                torch.nn.Conv3d(in_channels=hidden_C, out_channels=hidden_C, kernel_size=(1,2,2), stride=(1,2,2)),
                torch.nn.LeakyReLU()])
        
        # decide
        self.foreground_decider = torch.nn.Conv1d(in_channels=hidden_C, out_channels=1, kernel_size=1)
    
    @staticmethod
    def is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0

    def forward(self, x: list[Tensor]) -> Tensor:
        assert x[0].ndim == 5, "Input tensor should be 5D tensor, [B, C, Z, H, W], BUT got {}".format(x[0].shape)
        assert len(x) == len(self.embed_dims), "Input tensor should have {} layers, BUT got {}".format(len(self.embed_dims), len(x))
        
        # NOTE Detach from the encoder, not influencing the encoder, improve stability.
        x = [layer.detach() for layer in x]
        
        # Upsample align, feat: [input_B, input_C, input_Z, hidden_feat_HW, hidden_feat_HW]
        feat = [
            torch.nn.functional.interpolate(
                layer, 
                size=(self.Z_lengths[0],
                      self.hidden_feat_HW,
                      self.hidden_feat_HW),
                mode='nearest'
            )
            for layer in x
        ]
        feat = torch.cat(feat, dim=1) # [input_B, hidden_C, input_Z, hidden_feat_HW, hidden_feat_HW]
        
        # extract, will not change shape of feat
        for layer in self.z_extractor:
            feat = layer(feat)
        
        # downsample on Y X
        for layer in self.downsample:
            feat = layer(feat)
        feat = torch.nn.functional.max_pool3d(feat, kernel_size=(1, feat.size(3), feat.size(4))) # [B, C, Z, 1, 1]
        feat = feat.squeeze(-1).squeeze(-1) # [B, C, Z]
        
        # decider, [B, 1, Z]
        if self.use_checkpoint:
            feat = torch.utils.checkpoint.checkpoint(self.foreground_decider, feat)
        else:
            feat = self.foreground_decider(feat)
        
        return feat.squeeze(1) # [B, Z]

    def loss(
        self,
        inputs: tuple[Tensor],
        batch_data_samples: list[Seg3DDataSample],
        train_cfg:dict|None=None,
    ) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]):
                List of multi-level img features.
                (N, C, Z, Y, X)

            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.

            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        z_results = torch.sigmoid(self.forward(inputs)) # [B, Z]
        seg_label = self._stack_batch_gt(batch_data_samples, "gt_sem_seg") # [B, 1, Z, Y, X]
        foreground_Zs = torch.any(seg_label, dim=(1,3,4)).float() # [B, Z]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z_results, foreground_Zs)
        return {"loss_L3": loss * self.loss_weight}
