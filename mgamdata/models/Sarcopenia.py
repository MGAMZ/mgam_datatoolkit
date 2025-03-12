import pdb

import numpy as np
import torch
from torch import Tensor

from mmengine.model import BaseModule
from ..mm.mmseg_Dev3D import BaseDecodeHead_3D, Seg3DDataSample


class L3LocationDecoder(BaseDecodeHead_3D):
    def __init__(self, embed_dims:list[int], Z_lengths:list[int], threshold:float, loss_weight:float=1., *args, **kwargs):
        assert len(embed_dims) == len(Z_lengths)
        assert all([self.is_power_of_two(i) for i in Z_lengths]), "Z_downsample_ratio_from_backbone only support power of 2 for now."
        super().__init__(in_channels=embed_dims, 
                         channels=1, 
                         num_classes=1, 
                         in_index=list(range(len(embed_dims))),
                         input_transform='multiple_select', 
                         *args, **kwargs)
        self.embed_dims = embed_dims
        self.threshold = threshold
        self.loss_weight = loss_weight

        source_z = Z_lengths[0]
        source_c = embed_dims[0]
        for layer_idx, (z, c) in enumerate(zip(Z_lengths[1:], embed_dims[1:])):
            setattr(self, f"Z_upsample_{layer_idx}", 
                torch.nn.ConvTranspose1d(in_channels=c, 
                                         out_channels=source_c, 
                                         kernel_size=source_z//z, 
                                         stride=source_z//z))
        
        hidden_c = source_c*len(embed_dims)
        self.z_extractor = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=source_c*len(embed_dims), out_channels=source_c*len(embed_dims), kernel_size=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=source_c*len(embed_dims), out_channels=hidden_c, kernel_size=1),
            torch.nn.LeakyReLU()])
        for _ in range(18):
            self.z_extractor.extend([
                torch.nn.Conv1d(in_channels=hidden_c, out_channels=hidden_c, kernel_size=5, padding=2),
                torch.nn.LeakyReLU()])
        
        self.foreground_decider = torch.nn.Conv1d(in_channels=hidden_c, out_channels=1, kernel_size=1)
    
    @staticmethod
    def is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0

    def forward(self, x: tuple[Tensor]) -> Tensor:
        assert x[0].ndim == 5, "Input tensor should be 5D tensor, [B, C, Z, H, W], BUT got {}".format(x[0].shape)
        assert len(x) == len(self.embed_dims), "Input tensor should have {} layers, BUT got {}".format(len(self.embed_dims), len(x))
        
        # Upsample
        feat_aligned_shape = [
            getattr(self, f"Z_upsample_{layer_idx}")(layer_feat.mean(dim=(-1,-2))) 
            for layer_idx, layer_feat in enumerate(x[1:])
        ]
        feat = torch.cat([x[0].mean(dim=(-1,-2)), *feat_aligned_shape], dim=1) # [B, hidden_C*layers, Z]
        
        # extract
        for layer in self.z_extractor:
            feat = layer(feat)
        # [B, hidden_C, Z]
        
        # decider
        return self.foreground_decider(feat).squeeze(1) # [B, Z]

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
        loss = torch.nn.functional.l1_loss(z_results, foreground_Zs) * self.loss_weight
        return {"loss_L3": loss}
