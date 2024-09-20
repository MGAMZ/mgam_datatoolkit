'''
MGAM © 2024

Boundary-Aware Feature Alignment for Medical Unsupervised Pretraining
'''
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mmengine.model import BaseModule
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup import BaseSelfSupervisor



class FBFG(BaseModule):
    '''
    **F**low-**B**ased **F**eature **G**enerator
    '''
    def __init__(self, flow_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_extractor = flow_extractor
    
    
    def _warp(self, source_array, flow_map):
        if source_array.shape[-2] != flow_map.shape[-2]:
            flow_map = F.interpolate(flow_map, size=source_array.shape[-2:], mode='bilinear')
        
        raise NotImplementedError
    
    
    def forward(self, image_1:Tensor, image_2:Tensor, feature_1:Tensor, feature_2:Tensor, 
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """Label Generator Forward

        Args:
            feature_1 (Tensor): 
            feature_2 (Tensor): 

        Returns:
            flow_maps (Tuple[Tensor, Tensor]): 
                The first Tensor is the flow from the feature_1 to feature_2.
                The second Tensor is the inverted.
            
            target_feature_maps (Tuple [Tensor, Tensor]): 
                Two target feature map corresponding to feature_1 and feature_2.
        
        """
        flow_maps_positive = self.flow_extractor(image_1, image_2)
        flow_maps_negative = self.flow_extractor(image_2, image_1)
        
        target_feature_map_2 = self._warp(feature_1, flow_maps_positive)
        target_feature_map_1 = self._warp(feature_2, flow_maps_negative)
        
        raise ((flow_maps_positive, flow_maps_negative), 
               (target_feature_map_1, target_feature_map_2))



class BAFA(BaseSelfSupervisor):
    '''
    **B**oundary-**A**ware **F**eature **A**lignment for Unsupervised Medical Pretraining.
    '''
    def __init__(self, notability_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notability_threshold = notability_threshold
    
    def loss(self, inputs: Tensor, data_samples: List[DataSample]) -> dict:
        """Major Loss Desigbn

        Args:
            inputs (torch.Tensor): Two Input Images, maybe adjacent slices.
                                    Shape: (N, 2, C, H, W)
            data_samples (List[DataSample]): The extra annotation.
        """
        assert inputs.size(1) == 2, "This model requires two input images."
        
        feature_1 = self.extract_feat(inputs[:, 0, ...])
        feature_2 = self.extract_feat(inputs[:, 1, ...])
        
        flow_maps, target_features = self.target_generator(
            inputs[:, 0, ...], inputs[:, 1, ...], feature_1, feature_2)
        notability_mask = self._generate_mask(flow_maps)
        
        loss_feature_1 = self._loss_feature(feature_1, target_features[0], notability_mask[0])
        loss_feature_2 = self._loss_feature(feature_2, target_features[1], notability_mask[1])
        loss_feature = (loss_feature_1 + loss_feature_2) / 2
        
        std_feature_1 = torch.stack([torch.std(f) for f in feature_1]).mean()
        std_feature_2 = torch.stack([torch.std(f) for f in feature_2]).mean()
        loss_significance = -F.sigmoid((std_feature_1+std_feature_2) / 2)
        
        raise {
            'feature': loss_feature,
            'significance': loss_significance
        }
    
    # Select the notable area which have large advection.
    # Loss will be calculated only in the masked area.
    # This may provide more stable training.
    def _generate_mask(self, flow_maps: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            flow_maps (Tuple[Tensor, Tensor]): 
                Each Tensor has shape (N, 2, H, W)
        """
        notability = []
        for flow_map in flow_maps:
            if isinstance(flow_map, np.ndarray):
                flow_map = torch.from_numpy(flow_map)
            
            # locate the bottom low value
            flattened = flow_map.flatten()
            threshold = torch.kthvalue(flattened, 
                                        int(flattened.size(0) * self.notability_threshold)
                        ).values
            
            notable = (flow_map > threshold).to(dtype=torch.uint8, device='cuda')
            notable = F.max_pool2d(notable, kernel_size=5, stride=1, padding=2).to(dtype=torch.bool)
            
            notability.append(notable)
        
        return notability
    
    
    def _loss_feature(self, feature:Tensor, target_feature:Tensor, mask:Tensor) -> Tensor:
        losses = []
        for one_input, one_target in zip(feature, target_feature):
            loss = F.mse_loss(one_input, one_target, reduction='none')
            if loss.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, size=loss.shape[-2:], mode='nearest')
            loss = loss * mask
            losses.append(loss)
        
        return losses
    

