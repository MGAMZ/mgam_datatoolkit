import os
import argparse
import multiprocessing
import pdb
from tqdm import tqdm

import cv2
import pandas as pd
import numpy as np
import SimpleITK as sitk

from mgamdata.dataset.Totalsegmentator import META_CSV_PATH, DATA_ROOT
from mgamdata.dataset.Totalsegmentator.mm_dataset import TotalsegmentatorIndexer
from mgamdata.io.sitk_toolkit import split_image_label_pairs_to_2d



def process_case(args):
    case, source_dir, target_img_dir, target_ann_dir = args
    """处理单个案例的文件复制"""
    try:
        case_path = os.path.join(source_dir, case)
        ct_path = os.path.join(case_path, 'ct.mha')
        segmentation_path = os.path.join(case_path, 'segmentations.mha')
        
        # 读取数据
        ct_itk_image = sitk.ReadImage(ct_path)
        label_itk_image = sitk.ReadImage(segmentation_path)
        
        # 2D拆分
        fetcher_2D_slices = split_image_label_pairs_to_2d(ct_itk_image, label_itk_image)
        # 保存为tiff
        for i, (img, ann) in enumerate(fetcher_2D_slices):
            img_path = os.path.join(target_img_dir, case, f"{i:03d}.tiff")
            ann_path = os.path.join(target_ann_dir, case, f"{i:03d}.tiff")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            os.makedirs(os.path.dirname(ann_path), exist_ok=True)
            cv2.imwrite(img_path, img.astype(np.int16), [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_LZW])
            cv2.imwrite(ann_path, ann.astype(np.uint8), [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_LZW])

    except Exception as e:
        raise RuntimeError(f"Failed to process case {case}: {e}")



def generate_task_args(source_dir:str, target_dir:str, metainfo:pd.DataFrame):
    task_args = []
    for case in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, case)):
            try:
                split:str = metainfo.loc[case]['split']
            except:
                pdb.set_trace()
            target_img_dir = os.path.join(target_dir, 'img_dir', split)
            target_ann_dir = os.path.join(target_dir, 'ann_dir', split)
            
            one_task = (case, source_dir, target_img_dir, target_ann_dir)
            task_args.append(one_task)
            
    return task_args


def split_slices(source_dir, target_dir, use_multiprocessing):
    metainfo = pd.read_csv(META_CSV_PATH, index_col='image_id')
    task_args = generate_task_args(source_dir, target_dir, metainfo)
    
    if use_multiprocessing:
        with multiprocessing.Pool(24) as pool:
            fetcher = pool.imap_unordered(process_case, task_args)
            for _ in tqdm(
                    iterable=fetcher,
                    total=len(task_args),
                    desc="Processing cases",
                    dynamic_ncols=True,
                    leave=False):
                pass
    
    else:
        for task in tqdm(task_args, desc="Processing cases"):
            process_case(task)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将数据集转换为openmm目录结构")
    parser.add_argument('source_dir', type=str, help="源数据集目录")
    parser.add_argument('target_dir', type=str, help="目标数据集目录")
    parser.add_argument('--mp', action='store_true', help="使用多进程处理")
    
    args = parser.parse_args()
    
    split_slices(args.source_dir, args.target_dir, args.mp)
    # generate indexing json file
    TotalsegmentatorIndexer(os.path.join(args.target_dir))