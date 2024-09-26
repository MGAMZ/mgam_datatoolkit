import cv2
import numpy as np
import SimpleITK as sitk

from mmcv.transforms import BaseTransform
from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing_v2, sitk_resample_to_size


'''
NOTE 
规范化：在进入神经网络之前，
所有预处理的对外特性都应当遵循
[Z,Y,X]或[D,H,W]的维度定义
'''


class LoadImgFromOpenCV(BaseTransform):
    '''
    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape
    '''
    
    def transform(self, results: dict) -> dict:
        img_path = results['img_path']
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        results['img'] = img
        results['img_shape'] = img.shape[-2:]
        results['ori_shape'] = img.shape[-2:]
        return results



class LoadAnnoFromOpenCV(BaseTransform):
    '''
    Required Keys:

    - seg_map_path

    Modified Keys:

    - gt_seg_map
    - seg_fields
    '''
    def transform(self, results: dict) -> dict:
        if 'seg_map_path' in results:
            mask_path = results['seg_map_path']
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            results['gt_seg_map'] = mask
            results['seg_fields'].append('gt_seg_map')
        return results



class LoadFromMHA(BaseTransform):
    def __init__(self, resample_spacing=None, resample_size=None):
        assert not ((resample_spacing is not None) and (resample_size is not None))
        self.resample_spacing = resample_spacing
        self.resample_size = resample_size


    def _process_mha(self, mha, field):
        if self.resample_spacing is not None:
            mha = sitk_resample_to_spacing_v2(mha, self.spacing, field)
        if self.resample_size is not None:
            mha = sitk_resample_to_size(mha, self.resample_size, field)
        # mha.GetSize(): [H, W, D]
        mha_array = sitk.GetArrayFromImage(mha) # [D, W, H]
        return mha_array



class LoadImageFromMHA(LoadFromMHA):
    '''
    Required Keys:

    - img_path

    Modified Keys:

    - img
    - sitk_image
    '''
    def transform(self, results):
        img_path = results['img_path']
        img_mha = sitk.ReadImage(img_path)
        img = self._process_mha(img_mha, 'image')
        
        results['img'] = img  # output: [H, W, D]
        results['sitk_image'] = img_mha
        return results



class LoadMaskFromMHA(LoadFromMHA):
    '''
    Required Keys:

    - label_path
    - sitk_image

    Modified Keys:

    - gt_seg_map_index
    - gt_seg_map_channel
    '''
    @staticmethod
    def _split_channel(mask:np.ndarray):
        class_idxs = np.unique(mask)
        mask_channel = np.stack([mask==class_id for class_id in class_idxs], axis=-4)
        return mask_channel
    
    def transform(self, results):
        mask_path = results['label_path']
        mask_mha = sitk.ReadImage(mask_path)
        mask_mha.CopyInformation(results['sitk_image'])
        mask = self._process_mha(mask_mha, 'mask')
        
        results['gt_seg_map_index'] = mask # output: [H, W, D]
        results['gt_seg_map_channel'] = self._split_channel(mask)
        return results
