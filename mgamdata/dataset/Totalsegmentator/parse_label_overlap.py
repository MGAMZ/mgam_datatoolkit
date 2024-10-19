import os
import pdb
import multiprocessing as mp
from tqdm import tqdm

import numpy as np
import nibabel as nib

from mgamdata.dataset.Totalsegmentator.meta import DATA_ROOT



def load_case_data(case_name):
    case_path = os.path.join(DATA_ROOT, case_name)
    if not os.path.exists(case_path):
        raise FileNotFoundError(f"Case path {case_path} does not exist.")
    
    # 读取ct扫描
    ct_path = os.path.join(case_path, 'ct.nii.gz')
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT scan file {ct_path} does not exist.")
    ct_scan = nib.load(ct_path).get_fdata().astype(np.int16)
    
    # 读取所有分割文件
    segmentations_path = os.path.join(case_path, 'segmentations')
    if not os.path.exists(segmentations_path):
        raise FileNotFoundError(
            f"Segmentations path {segmentations_path} does not exist.")
    
    segmentations = {}
    for filename in os.listdir(segmentations_path):
        if filename.endswith('.nii.gz'):
            class_name = filename.split('.')[0]
            file_path = os.path.join(segmentations_path, filename)
            segmentations[class_name] = nib.load(file_path).get_fdata().astype(np.uint8)
    
    return {
        'ct_scan': ct_scan,
        'segmentations': segmentations
    }



def process_segmentation(args):
    class_name, segmentation = args
    mask = (segmentation > 0).astype(np.uint8)
    return mask, class_name



def calculate_overlap(segmentations):
    # 获取所有分割数组的形状，假设所有分割数组形状相同
    shape = next(iter(segmentations.values())).shape
    
    # 初始化重叠矩阵和重叠类字典
    overlap_matrix = np.zeros(shape, dtype=np.uint8)
    overlap_classes = np.empty(shape, dtype=object)
    for index, _ in np.ndenumerate(overlap_classes):
        overlap_classes[index] = []
    
    # 遍历每个分割数组
    for class_name, segmentation in tqdm(
            segmentations.items(),
            desc="Calculating overlap",
            total=len(segmentations),
            dynamic_ncols=True,
            leave=False):
        # 对于每个体素位置，如果该位置有标注，则在重叠矩阵中相应位置的值加1，并记录类名
        mask = (segmentation > 0).astype(np.uint8)
        overlap_matrix += mask
        for index in np.ndindex(mask.shape):
            if mask[index]:
                overlap_classes[index].append(class_name)
    
    return overlap_matrix, overlap_classes



if __name__ == '__main__':
    sample = load_case_data('s0000')
    overlap_matrix = calculate_overlap(sample['segmentations'])
    pdb.set_trace()
