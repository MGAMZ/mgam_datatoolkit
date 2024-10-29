import os
import pdb

import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk




def convert_nii_sitk(nii_path:str, 
                     dtype=np.float32, 
                     nii_fdata_order:str='zyx') -> sitk.Image:
    """保证itk mha格式维度顺序为zyx"""
    
    nib_img = nib.load(nii_path)
    nib_array:np.ndarray = nib_img.get_fdata()
    nib_meta = nib_img.header
    nib_spacing = nib_meta['pixdim'][1:4].tolist() # type: ignore
    nib_origin = nib_meta.get_qform()[0:3, 3].tolist()
    nib_direction = nib_meta.get_qform()[0:3, 0:3].flatten().tolist()
    
    if nii_fdata_order == 'xyz':
        nib_array = np.transpose(nib_array, (2, 1, 0))
    elif nii_fdata_order == 'zyx':
        nib_spacing = nib_spacing[::-1]
        nib_origin = nib_origin[::-1]
        nib_direction = nib_direction[::-1]
    else:
        raise ValueError(f"Invalid nii_fdata_order: {nii_fdata_order}")
    
    sitk_img = sitk.GetImageFromArray(nib_array.astype(dtype))
    sitk_img.SetSpacing(nib_spacing)
    sitk_img.SetOrigin(nib_origin)
    sitk_img.SetDirection(nib_direction)
    return sitk_img



def merge_masks(nii_paths: list[str], 
                class_index_map: dict[str, int],
                dtype=np.uint8
                ) -> np.ndarray:
    """
    将所有类的掩码合并到一个掩码中并返回NumPy数组。
    
    :param nii_paths: 所有nii文件的路径列表
    :param class_index_map: 类名到索引的映射字典
    :return: 合并后的NumPy数组
    """
    # 初始化一个空的掩码图像
    merged_mask = None
    
    # 遍历nii文件路径列表中的每个文件
    for seg_file_path in nii_paths:
        if os.path.isfile(seg_file_path):
            assert seg_file_path.endswith('.nii.gz')
            class_name = os.path.basename(seg_file_path)[:-7]
            class_index = class_index_map.get(class_name)
            if class_index is None:
                raise ValueError(f"Class name {class_name} not found in class_index_map: {seg_file_path}")
            
            # 读取掩码文件
            mask = nib.load(seg_file_path)
            mask_array = mask.get_fdata()
            # 初始化合并掩码
            if merged_mask is None:
                merged_mask = np.zeros_like(mask_array, dtype=dtype)
            # 将当前类的掩码添加到合并掩码中
            merged_mask[mask_array == 1] = class_index
    
    if merged_mask is None:
        raise ValueError("No mask found in the provided paths")
    
    return merged_mask.astype(dtype)



if __name__ == "__main__":
    nii_path = '/fileser51/zhangyiqin.sx/Totalsegmentator_Data/Totalsegmentator_dataset_v201/s0000/ct.nii.gz'
    sitk_img = convert_nii_sitk(nii_path)