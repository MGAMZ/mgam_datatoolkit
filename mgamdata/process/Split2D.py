import os
import pdb
import argparse
import json
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

import cv2
import numpy as np
import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import split_image_label_pairs_to_2d


IMAGE_DTYPE = np.int16
LABEL_DTYPE = np.uint8


def get_series_uids(input_folder):
    """获取所有可用的SeriesUID列表"""
    image_folder = os.path.join(input_folder, "image")
    series_uids = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".mha"):
            series_uid = os.path.splitext(filename)[0]
            if os.path.exists(os.path.join(input_folder, "label", f"{series_uid}.mha")):
                series_uids.append(series_uid)
    return series_uids


def process_single_series(series_uid, 
                          input_folder, 
                          out_folder, 
                          resize:list[int]|None=None, 
                          foreground_ratio:float|None=None):
    """处理单个Series的3D MHA文件，将其分割为2D切片并保存"""
    # 校验
    image_path = os.path.join(input_folder, "image", f"{series_uid}.mha")
    label_path = os.path.join(input_folder, "label", f"{series_uid}.mha")
    image_out_dir = os.path.join(out_folder, "image", series_uid)
    label_out_dir = os.path.join(out_folder, "label", series_uid)
    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print(f"Warning: Missing files for SeriesUID {series_uid}")
        return False
    
    # 读取，创建目录
    try:
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        image = sitk.DICOMOrient(image, 'LPI')
        label = sitk.DICOMOrient(label, 'LPI')
    except Exception as e:
        print(f"Error reading files for SeriesUID {series_uid}: {e}")
        return False
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    
    # 切分并保存2D切片
    for idx, (image_slice, label_slice) in enumerate(split_image_label_pairs_to_2d(image, label)):
        # 如果只处理前景切片，且当前切片没有前景，则跳过
        if foreground_ratio is not None:
            r = ((label_slice > 0).astype(int).sum() / label_slice.size).astype(float)
            if r < foreground_ratio:
                continue
        # 可选缩放
        if resize:
            image_slice = cv2.resize(image_slice, resize[::-1], interpolation=cv2.INTER_CUBIC)
            label_slice = cv2.resize(label_slice, resize[::-1], interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(image_out_dir, f"{idx}.tiff"), image_slice.astype(IMAGE_DTYPE), [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_LZW])
        cv2.imwrite(os.path.join(label_out_dir, f"{idx}.tiff"), label_slice.astype(LABEL_DTYPE), [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_LZW])
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Split 3D MHA files into 2D slices")
    parser.add_argument("input_folder", type=str, help="Input folder containing 'image' and 'label' subfolders with MHA files")
    parser.add_argument("out_folder", type=str, help="Output folder to save 2D slices")
    parser.add_argument("--mp", action="store_true", help="Use multiprocessing for faster processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to use")
    parser.add_argument("--size", type=int, nargs=2, default=None, help="Resize output images to this size (Y, X)")
    parser.add_argument("--foreground-ratio", type=float, default=None, help="Only process foreground slices")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {args.input_folder}")
    os.makedirs(args.out_folder, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_folder, "SplitLog.json"), "w"), indent=4)
    
    # 获取所有SeriesUID
    series_uids = get_series_uids(args.input_folder)
    print(f"Found {len(series_uids)} series to process.")
    if not series_uids:
        print("No valid series found. Exiting.")
        return
    
    # 处理所有Series
    process_func = partial(process_single_series,
                           input_folder=args.input_folder,
                           out_folder=args.out_folder,
                           resize=args.size,
                           foreground_ratio=args.foreground_ratio)
    if args.mp:
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, series_uids),
                total=len(series_uids),
                desc="Split2D",
                dynamic_ncols=True,
            ))
    else:
        results = []
        for series_uid in tqdm(series_uids, desc="Processing Series"):
            results.append(process_func(series_uid))
    
    successful = sum(1 for r in results if r)
    print(f"Successfully processed {successful} out of {len(series_uids)} series.")


if __name__ == "__main__":
    main()
