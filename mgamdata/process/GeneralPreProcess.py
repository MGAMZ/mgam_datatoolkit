import os
import random
from typing import List, Dict, Sequence, Union
from typing_extensions import deprecated

import numpy as np

from mmcv.transforms import BaseTransform


'''
NOTE 
规范化：在进入神经网络之前，
所有预处理的对外特性都应当遵循
[Z,Y,X]或[D,H,W]的维度定义
'''



class CropSlice_Foreground(BaseTransform):
    '''
    Required Keys:
    
    - img
    - gt_seg_map_index
    - gt_seg_map_channel

    Modified Keys:

    - img
    - gt_seg_map_index
    - gt_seg_map_channel
    '''
    def __init__(self, num_slices:int, ratio:float=0.9):
        self.num_slices = num_slices
        self.ratio = ratio # 一定几率下，本处理才会生效
        
    def _locate_possible_start_slice_with_non_background(self, mask):
        assert mask.ndim == 3, f"Invalid Mask Shape: Expected [D,H,W], but got {mask.shape}"
        # locate non-background slices
        slices_not_pure_background = np.argwhere(np.any(mask, axis=(1,2)))
        start_slice, end_slice = slices_not_pure_background.min(), slices_not_pure_background.max()
        non_background_slices = np.arange(start_slice, end_slice, dtype=np.uint32)
        
        # locate the range of possible start slice, 
        # which could ensure the selected slices are not all background
        min_possible_start_slice = max(0, non_background_slices[0] - self.num_slices + 1)
        max_possible_start_slice = max(0, min(non_background_slices[-1], mask.shape[0] - self.num_slices))
        
        return (min_possible_start_slice, max_possible_start_slice)
    
    def transform(self, results):
        if np.random.rand(1) > self.ratio:
            return results
        
        mask = results['gt_seg_map_index']
        min_start_slice, max_start_slice = self._locate_possible_start_slice_with_non_background(mask)
        selected_slices = np.arange(min_start_slice, max_start_slice + self.num_slices - 1)

        results['img'] = np.take(results['img'], selected_slices, axis=0)
        results['gt_seg_map_channel'] = np.take(results['gt_seg_map_channel'], selected_slices, axis=1)
        results['gt_seg_map_index'] = np.take(results['gt_seg_map_index'], selected_slices, axis=0)
        return results



class WindowSet(BaseTransform):
    '''
    Required Keys:

    - img

    Modified Keys:

    - img
    '''
    def __init__(self, location, width):
        self.clip_range = (location - width//2, location + width//2)
        self.location = location
        self.width = width

    def _window_norm(self, img: np.ndarray):
        img = np.clip(img, self.clip_range[0], self.clip_range[1])  # Window Clip
        img = img - self.clip_range[0]  # HU bias to positive
        img = img / self.width # Zero-One Normalization
        return img.astype(np.float32)

    def transform(self, results):
        results['img'] = self._window_norm(results['img'])
        return results



class TypeConvert(BaseTransform):
    '''
    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map
    '''
    def transform(self, results):
        if 'img' in results:
            results['img'] = results['img'].astype(np.float32)
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'].astype(np.uint8)
        return results



class RandomRoll(BaseTransform):
    '''
    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map
    '''
    def __init__(self, 
                 direction: str|list[str],
                 gap: float|list[float],
                 erase: bool=False,
                 pad_val: int=0,
                 seg_pad_val: int=0):
        """
        direction和prob是对应的，可指定随机向多个方向roll
        
        :param direction: roll的方向, horizontal 或者 vertical
        :param gap: 对应方向的最大roll距离
        :param erase: 是否擦除roll后的部分
        :param pad_val: 擦除的填充值
        :param seg_pad_val: seg擦除的填充值
        """
        if isinstance(direction, str):
            direction = [direction]
        if isinstance(gap, float):
            gap = [gap]
        assert len(direction) == len(gap), "所有参数的长度必须相同" # type: ignore
        
        self.direction = direction
        self.gap = {k:v for k,v in zip(direction, gap)} # type: ignore
        self.erase = erase
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
    
    @staticmethod
    def _roll(results, gap, axis):
        results['img'] = np.roll(results['img'], shift=gap, axis=axis)
        results['gt_seg_map'] = np.roll(results['gt_seg_map'], shift=gap, axis=axis)
        return results
    
    
    def _erase(self, results, gap, axis):
        if axis == -1:
            if gap > 0:
                results['img'][..., :gap] = self.pad_val
                results['gt_seg_map'][..., :gap] = self.seg_pad_val
            else:
                results['img'][..., gap:] = self.pad_val
                results['gt_seg_map'][..., gap:] = self.seg_pad_val
        
        if axis == -2:
            if gap > 0:
                results['img'][..., :gap, :] = self.pad_val
                results['gt_seg_map'][..., :gap, :] = self.seg_pad_val
            else:
                results['img'][..., gap:, :] = self.pad_val
                results['gt_seg_map'][..., gap:, :] = self.seg_pad_val
        if axis == -3:
            if gap > 0:
                results['img'][..., :gap, :, :] = self.pad_val
                results['gt_seg_map'][..., :gap, :, :] = self.seg_pad_val
            else:
                results['img'][..., gap:, :, :] = self.pad_val
                results['gt_seg_map'][..., gap:, :, :] = self.seg_pad_val
        return results
    
    
    def transform(self, results):
        if 'horizontal' in self.direction:
            axis = -2
            gap = random.randint(-self.gap['horizontal'], self.gap['horizontal'])
            results = self._roll(results, gap, axis)
        if 'vertical' in self.direction:
            axis = -1
            gap = random.randint(-self.gap['vertical'], self.gap['vertical'])
            results = self._roll(results, gap, axis)
        if 'normal' in self.direction:
            axis = -3
            gap = random.randint(-self.gap['normal'], self.gap['normal'])
            results = self._roll(results, gap, axis)
        
        return results



class InstanceNorm(BaseTransform):
    '''
    Required Keys:

    - img

    Modified Keys:

    - img
    '''
    def __init__(self, eps:float=1e-3) -> None:
        super().__init__()
        self.eps = eps
    
    def transform(self, results):
        ori_dtype = results['img'].dtype
        results['img'] = (results['img'] - results['img'].mean())
        results['img'] = results['img'] / (results['img'].std() + self.eps)
        results['img'] = results['img'].astype(ori_dtype)
        return results
