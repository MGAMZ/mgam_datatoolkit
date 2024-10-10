import os
import argparse
import multiprocessing
from tqdm import tqdm

import cv2
import numpy as np
import SimpleITK as sitk

from mgamdata.dataset.Totalsegmentator import CLASS_INDEX_MAP
from mgamdata.io.sitk_toolkit import merge_masks, split_image_label_pairs_to_2d




def create_directory_structure(base_dir):
    """创建目标目录结构"""
    dirs = [
        os.path.join(base_dir, 'img_dir', 'train'),
        os.path.join(base_dir, 'img_dir', 'val'),
        os.path.join(base_dir, 'ann_dir', 'train'),
        os.path.join(base_dir, 'ann_dir', 'val')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)



def process_case(args):
    case, source_dir, img_train_dir, ann_train_dir = args
    """处理单个案例的文件复制"""
    try:
        case_path = os.path.join(source_dir, case)
        ct_path = os.path.join(case_path, 'ct.mha')
        segmentation_path = os.path.join(case_path, 'segmentations')
        if not os.path.exists(ct_path):
            return
        
        ct_itk_image = sitk.ReadImage(ct_path)
        
        # 融合独立的annotation
        label_itk_image = merge_masks(
            mha_paths=[os.path.join(segmentation_path, file) 
                       for file in os.listdir(segmentation_path)
                       if file.endswith('.mha')],
            class_index_map=CLASS_INDEX_MAP
        )
        # 拆分成2D图像
        fetcher = split_image_label_pairs_to_2d(ct_itk_image, label_itk_image)
        
        # 保存为tiff
        for i, (img, ann) in enumerate(fetcher):
            img_path = os.path.join(img_train_dir, f"{case}_{i:03d}.tiff")
            ann_path = os.path.join(ann_train_dir, f"{case}_{i:03d}.tiff")
            cv2.imwrite(img_path, img.astype(np.float32), [cv2.IMWRITE_TIFF_COMPRESSION, 5])
            cv2.imwrite(ann_path, ann.astype(np.uint8), [cv2.IMWRITE_TIFF_COMPRESSION, 5])

    except Exception as e:
        raise RuntimeError(f"Failed to process case {case}: {e}")



def main(source_dir, target_dir, use_multiprocessing):
    # 创建目标目录结构
    create_directory_structure(target_dir)
    
    img_train_dir = os.path.join(target_dir, 'img_dir', 'train')
    ann_train_dir = os.path.join(target_dir, 'ann_dir', 'train')
    
    cases = [case for case in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, case))]
    
    if use_multiprocessing:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            for _ in tqdm(
                    iterable=pool.imap_unordered(
                        func=process_case, 
                        iterable=[(case, source_dir, img_train_dir, ann_train_dir) 
                                  for case in cases],
                        chunksize=16),
                    total=len(cases),
                    desc="Processing cases",
                    dynamic_ncols=True,
                    leave=False):
                pass
    
    else:
        for case in tqdm(cases, desc="Processing cases"):
            process_case(case, source_dir, img_train_dir, ann_train_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将数据集转换为openmm目录结构")
    parser.add_argument('source_dir', type=str, help="源数据集目录")
    parser.add_argument('target_dir', type=str, help="目标数据集目录")
    parser.add_argument('--mp', action='store_true', help="使用多进程处理")
    
    args = parser.parse_args()
    
    main(args.source_dir, args.target_dir, args.mp)