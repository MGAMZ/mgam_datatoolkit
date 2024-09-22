'''
MGAM © 2024

Boundary-Aware Feature Alignment for Medical Unsupervised Pretraining
'''
from abc import abstractmethod
from typing import List, Dict, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import functional as F
from scipy.ndimage import maximum_filter

from mmcv.transforms import BaseTransform
from mmengine.model import BaseModule
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup import BaseSelfSupervisor





class OpticalFlowGenerator(BaseTransform):
    '''
    Select the notable area which have large advection.
    Loss will be calculated only in the masked area.
    This may provide more stable training.
    '''
    def __init__(self, notability_level=0.1):
        """
        Args:
            notability_level (float, optional): 
                Higher value leads to fewer activated flow. Range from 0 to 1.
        """
        self.notability_level = notability_level

    @abstractmethod
    def _calc_flow(self, refer_img:NDArray, target_img:NDArray) -> NDArray:
        """Calculate the optical flow between two images.
        
        Args:
            refer_img (ndarray): The first image. Shape [C, H, W].
            target_img (ndarray): The second image. Shape [C, H, W].
        """
        return NotImplementedError


    def _calc_notability(self, flow_map:NDArray) -> NDArray:
        # locate the bottom low value
        flattened = flow_map.flatten()
        kth = int(flattened.size(0) * self.notability_level)
        threshold = np.partition(flattened, kth)[kth]

        # max pooling
        notable = (flow_map > threshold).astype(np.uint8)
        notable = maximum_filter(notable, size=5)
        return notable


    def transform(self, results: Dict) -> Dict:
        """
        Required Keys:
            - img (ndarray): [2, C, H, W] two images.
        
        Added Keys:
            - flow_pos (ndarray): [H, W, 2] optical flow map.
            - flow_neg (ndarray): [H, W, 2] inverted optical flow map.
            - notability_pos (Tensor): The notable area of flow_pos.
            - notability_neg (Tensor): The notable area of flow_neg.
        """
        
        results['flow_pos'] = self._calc_flow(results['img'][0], results['img'][1])
        results['flow_neg'] = self._calc_flow(results['img'][1], results['img'][0])
        if self.notability_level is not None:
            results['notability_pos'] = self._calc_notability(results['flow_pos'])
            results['notability_neg'] = self._calc_notability(results['flow_neg'])
            
        raise results



class BroxOptFlowGenerator(OpticalFlowGenerator):
    def __init__(self, size, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.stride = stride
        self.pts = self._generate_pts(size, stride) # [N, 2]
        self.key_points_detector = cv2.ORB().create(nfeatures=25**2)

    @staticmethod
    def _generate_pts(size, stride):
        rows = np.arange(0, size[0], stride[0])
        cols = np.arange(0, size[1], stride[0])
        row_indices, col_indices = np.meshgrid(rows, cols, indexing='ij')
        coordinates = np.stack([row_indices.ravel(), col_indices.ravel()], axis=-1)
        return coordinates


    def _calc_flow(self, refer_img:NDArray, target_img:NDArray) -> NDArray:
        """https://blog.csdn.net/2301_77444219/article/details/139426280

        Args:
            refer_img (NDArray): [C, H, W] The first image.
            target_img (NDArray): [C, H, W] The second image.

        Returns:
            next_pts: The key points location in target image calculated by optical flow tracing algorithm.
        """
        if refer_img.ndim == 3:
            refer_img = cv2.cvtColor(refer_img.transpose(1,2,0), cv2.COLOR_RGB2GRAY)
        if target_img.ndim == 3:
            target_img = cv2.cvtColor(target_img.transpose(1,2,0), cv2.COLOR_RGB2GRAY)
        
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(refer_img, target_img, self.pts, self.pts)
        return next_pts



class FBFG(BaseModule):
    '''
    **F**low-**B**ased **F**eature **G**enerator
    '''
    def __init__(self, flow_extractor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_extractor = flow_extractor


    def _warp(self, features, flow_map):
        warpped_features = []
        
        for feature in features:
            if feature.shape[-2] != flow_map.shape[-2]:
                flow_map = F.interpolate(flow_map, size=feature.shape[-2:], mode='bilinear')
            warpped = F.grid_sample(feature, flow_map, mode='bilinear')
            warpped_features.append(warpped)

        return features


    def forward(self, 
                feature_1: List[Tensor],
                feature_2: List[Tensor],
                data_samples: List[DataSample]
                ) -> Tuple[List[Tensor], List[Tensor]]:
        """Label Generator Forward

        Required Annotations:
            - flow_pos (Tensor): The flow from the first image to the second. Shape [2, H, W].
            - flow_neg (Tensor): The flow from the second image to the first. Shape [2, H, W].

        Args:
            feature_1 (List[Tensor]): Feature maps of slice 1 from all layers, each has shape [B, C, H, W].
            feature_2 (List[Tensor]): Feature maps of slice 2 from all layers, each has shape [B, C, H, W].

        Returns:
            target_feature_maps (List[Tensor, Tensor]): 
                Two target feature map corresponding to feature_1 and feature_2.
        """
        batched_flow_pos = torch.stack([sample.flow_pos for sample in data_samples])
        batched_flow_neg = torch.stack([sample.flow_neg for sample in data_samples])
        target_feature_map_2 = self._warp(feature_1, batched_flow_pos)
        target_feature_map_1 = self._warp(feature_2, batched_flow_neg)

        return (target_feature_map_1, target_feature_map_2)



class BAFA(BaseSelfSupervisor):
    '''
    **B**oundary-**A**ware **F**eature **A**lignment for Unsupervised Medical Pretraining.
    '''
    def __init__(self, notability_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notability_threshold = notability_threshold


    def loss(self, inputs: Tensor, data_samples: List[DataSample]) -> dict:
        """Major Loss Design

        Args:
            inputs (torch.Tensor): Two Input Images, maybe adjacent slices.
                                    Shape: (N, 2, C, H, W)
            data_samples (List[DataSample]): The extra annotation.
                - flow_pos (Tensor): The flow from the first image to the second.
                - flow_neg (Tensor): The flow from the second image to the first.
                - notability_pos (Tensor): The notable area of flow_pos.
                - notability_neg (Tensor): The notable area of flow_neg.
        """
        assert inputs.size(1) == 2, "This model requires two input images."

        feature_1 = self.extract_feat(inputs[:, 0, ...])
        feature_2 = self.extract_feat(inputs[:, 1, ...])

        target_feature_1, target_feature_2 = self.target_generator(
            inputs[:, 0, ...], inputs[:, 1, ...], data_samples)

        batched_nota_mask_pos = torch.stack([sample.notability_pos for sample in data_samples])
        batched_nota_mask_neg = torch.stack([sample.notability_neg for sample in data_samples])

        loss_feature_1 = self._loss_feature(feature_1, target_feature_1, batched_nota_mask_pos)
        loss_feature_2 = self._loss_feature(feature_2, target_feature_2, batched_nota_mask_neg)
        loss_feature = (loss_feature_1 + loss_feature_2) / 2

        std_feature_1 = torch.stack([torch.std(f) for f in feature_1]).mean()
        std_feature_2 = torch.stack([torch.std(f) for f in feature_2]).mean()
        loss_significance = -F.sigmoid((std_feature_1+std_feature_2) / 2)

        return {
            'feature': loss_feature,
            'significance': loss_significance
        }


    def _loss_feature(self, feature:Tensor, target_feature:Tensor, mask:Tensor) -> Tensor:
        losses = []
        for one_input, one_target in zip(feature, target_feature):
            loss = F.mse_loss(one_input, one_target, reduction='none')
            if loss.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, size=loss.shape[-2:], mode='nearest')
            loss = loss * mask
            losses.append(loss)

        return losses
