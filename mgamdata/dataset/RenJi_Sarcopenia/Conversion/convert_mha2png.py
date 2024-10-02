import os
import os.path as osp
import pdb
import json
from typing import Tuple, List, Dict, Union, Sequence, Optional
from colorama import Fore, Style
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from pprint import pprint

import cv2
import numpy as np
import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing_v2
from mgamdata.dataset.RenJi_Sarcopenia.L3 import find_L3_slices


FOREGROUND_THRESHOLD = 0.1



def auto_recursive_search_for_mha_sample_pair(
        mha_file_root:str, target_save_root:str, spacing:Optional[Tuple]=None):
    task_list = []
    task_seriesUID = []
    for root, dirs, files in os.walk(mha_file_root):
        for file in files:
            label_path = Path(osp.join(root, file))
            if label_path.suffix == '.mha' and label_path.parent.name == 'label':
                label_relative_to_root = label_path.relative_to(mha_file_root)
                image_path = label_path.parent.parent / 'image' / label_path.name
                image_relative_to_root = image_path.relative_to(mha_file_root)
                
                image_target_path = osp.join(target_save_root, 
                                             image_relative_to_root.parent, 
                                             image_relative_to_root.stem, 
                                             image_relative_to_root.stem)
                label_target_path = osp.join(target_save_root, 
                                             label_relative_to_root.parent, 
                                             label_relative_to_root.stem, 
                                             label_relative_to_root.stem)
                
                task_list.append([(image_path, image_target_path), 
                                  (label_path, label_target_path), 
                                  spacing])
                task_seriesUID.append(Path(image_path).stem)
    
    foreground_slicess = find_L3_slices(task_seriesUID)
    for i in range(len(task_list)):
        task_list[i].append(foreground_slicess[i])
    
    return task_list


def check_task(target_save_root:str):
    file_paths = []
    for root, dirs, files in os.walk(target_save_root):
        for file in files:
            if file.endswith('.tiff'):
                file_paths.append(osp.join(root, file))
    
    return file_paths


def process_one(param: Tuple[Tuple[str, str], Tuple[str, str], np.ndarray, np.ndarray]):
    image_paths = param[0]
    label_paths = param[1]
    spacing:Tuple = param[2]
    L3_slices = param[3]
    
    try:
        # 载入mha
        image = sitk.ReadImage(image_paths[0])
        label = sitk.ReadImage(label_paths[0])
        # 可选spacing
        if spacing is not None:
            image = sitk_resample_to_spacing_v2(image, spacing, field='image')
            label = sitk_resample_to_spacing_v2(label, spacing, field='label')
        # 转为numpy
        image = sitk.GetArrayFromImage(image) # [D, H, W]
        label = sitk.GetArrayFromImage(label) # [D, H, W]
        
        # 选择非空部分
        foreground_slice = (len(image) - L3_slices)[::-1]
        image = image[foreground_slice[0]:foreground_slice[1]]
        label = label[foreground_slice[0]:foreground_slice[1]]
        
        # 检查有效标注区域面积是否足够
        # 在这一步就去除异常mask
        if FOREGROUND_THRESHOLD is not None:
            foreground_pixel = np.sum(label!=0, axis=(1, 2)) # [D]
            all_pixel = np.prod(label.shape[1:])
            valid_slice = foreground_pixel > (all_pixel * FOREGROUND_THRESHOLD)
            image = image[valid_slice]
            label = label[valid_slice]
        
        # slice-wise保存为png
        for slice_idx, (image_slice, label_slice) in enumerate(zip(image, label)):
            image_target_path = image_paths[1] + f'_{slice_idx}.tiff'
            label_target_path = label_paths[1] + f'_{slice_idx}.tiff'
            
            image_slice = image_slice.astype(np.float32)
            label_slice = label_slice.astype(np.uint8)
            
            if not osp.exists(image_target_path):
                os.makedirs(osp.dirname(image_paths[1]), exist_ok=True)
                cv2.imwrite(image_target_path, image_slice)
            if not osp.exists(label_target_path):
                os.makedirs(osp.dirname(label_paths[1]), exist_ok=True)
                cv2.imwrite(label_target_path, label_slice)
        
        return True
        
    except Exception as e:
        return {
            'failed_path': [str(image_paths), str(label_paths)],
            'error': str(e)
        }


def check_one(path:str) -> Union[bool, Dict]:
    try:
        read = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if read.shape[-2:] != (512, 512):
            return {
                'failed_path': str(path),
                'error': 'shape not match'
            }
        return True
        
    except Exception as e:
        return {
            'failed_path': str(path),
            'error': str(e)
        }


def convert(task_list: List):
    with Pool(32) as p:
        failed = []
        fetcher = p.imap_unordered(process_one, task_list, chunksize=8)
        
        for result in tqdm(fetcher, desc='Converting', leave=True, dynamic_ncols=True, total=len(task_list)):
            if result is not True:
                failed.append(result)
                
    return failed


def check(dest_root: str):
    tasks = check_task(dest_root)
    print("Found", len(tasks), "tiff files.")
    
    with Pool(32) as p:
        failed = []
        fetcher = p.imap_unordered(check_one, tasks, chunksize=8)
        
        for result in tqdm(fetcher, desc='Checking', leave=True, dynamic_ncols=True, total=len(tasks)):
            if isinstance(result, Dict):
                failed.append(result)
                os.remove(result['failed_path'])
        
    return failed


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Convert mha to tiff.')
    parser.add_argument('mha_file_root', type=str, help='Root of mha files.')
    parser.add_argument('dest_root', type=str, help='Root of tiff files.')
    parser.add_argument('--spacing', type=str, default=None, help='Spacing of the mha files.')
    return parser.parse_args()


if __name__ == '__main__':
    ''' 转换原有dcm序列为sitk.image MHA文件。将一同处理label。
        
        执行本脚本前应当先执行dcm2mha
        mha_file_root: 该目录下应当包括image和label两个子目录，均存放mha文件
        dest_root: 保存转换后的tiff文件, 将会在该目录下自动建立相同的目录结构
        
        NOTE 请注意本脚本顶端的阈值全局变量。该阈值代表被认为有效标注的最小面积占比。
    '''
    args = parse_args()
    
    task_list = auto_recursive_search_for_mha_sample_pair(args.mha_file_root, args.dest_root, args.spacing)
    print(f'Found {len(task_list)} pairs of mha files.')
    
    failed = []
    failed += convert(task_list)    # 执行转换
    failed += check(args.dest_root) # 检查图片
    
    os.makedirs(args.dest_root, exist_ok=True)
    json.dump(failed, 
                open(osp.join(args.dest_root, 'failed.json'), 'w'), 
                indent=4, 
                ensure_ascii=False)

    print('Failed Cases:')
    pprint(failed)
