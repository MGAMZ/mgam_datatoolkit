import os
import argparse
import re
import pdb
import multiprocessing
from tqdm import tqdm
from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk

from mgamdata.io.nii_toolkit import convert_nii_sitk, merge_masks
from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing_v2, sitk_resample_to_size, sitk_resample_to_image
from mgamdata.dataset.Totalsegmentator.meta import CLASS_INDEX_MAP



def maybe_skip(series_nii_image_path:str, series_nii_label_path:str):
    try:
        if os.path.exists(series_nii_image_path):
            sitk.ReadImage(series_nii_image_path)
        if os.path.exists(series_nii_label_path):
            sitk.ReadImage(series_nii_label_path)
    except Exception as e:
        return False
    
    return True

def convert_one_case(args):
    series_nii_image_path, series_output_folder, spacing, size = args
    output_image_folder = os.path.join(series_output_folder, 'image')
    output_label_folder = os.path.join(series_output_folder, 'label')
    series_nii_label_path = series_nii_image_path.replace('image', 'label').replace('_0000.nii.gz', '.nii.gz')
    
    # 构建路径，保持文件存储结构不变
    series_id = os.path.basename(series_nii_image_path).split('_')[1]
    output_image_mha_path = os.path.join(output_image_folder, f"{series_id}.mha")
    output_label_mha_path = os.path.join(output_label_folder, f"{series_id}.mha")
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    if os.path.exists(output_image_mha_path) and os.path.exists(output_label_mha_path):
        return
    
    # sitk读取nii
    input_image_mha = convert_nii_sitk(series_nii_image_path, dtype=np.int16, nii_fdata_order='xyz')
    if os.path.exists(series_nii_label_path):
        input_label_mha = convert_nii_sitk(series_nii_label_path, dtype=np.uint8, nii_fdata_order='xyz')
    
    # 可选根据spacing或size进行重采样
    if spacing is not None:
        assert size is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_spacing_v2(input_image_mha, spacing, 'image')
        if os.path.exists(series_nii_label_path):
            # input_label_mha = sitk_resample_to_spacing_v2(input_label_mha, spacing, 'label')
            input_label_mha = sitk_resample_to_image(input_label_mha, input_image_mha, 'label')
    if size is not None:
        assert spacing is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_size(input_image_mha, size, 'image')
        if os.path.exists(series_nii_label_path):
            input_label_mha = sitk_resample_to_size(input_label_mha, size, 'label')
    
    # 写入mha
    sitk.WriteImage(input_image_mha, output_image_mha_path, useCompression=True)
    if os.path.exists(series_nii_label_path):
        assert input_image_mha.GetSize() == input_label_mha.GetSize(), "Image and label size mismatch."
        sitk.WriteImage(input_label_mha, output_label_mha_path, useCompression=True)


def convert_and_save_nii_to_mha(input_dir: str, 
                                output_dir: str, 
                                use_mp: bool, 
                                spacing:Sequence[float|int]|None=None, 
                                size:Sequence[float|int]|None=None):
    task_list = []
    nii_image_dir = os.path.join(input_dir, 'image')
    for series_name in os.listdir(nii_image_dir):
        if series_name.endswith('.nii.gz'):
            series_nii_image_path = os.path.join(nii_image_dir, series_name)
            task_list.append((series_nii_image_path, output_dir, spacing, size))
    
    if use_mp:
        with multiprocessing.Pool() as pool:
            for _ in tqdm(
                pool.imap_unordered(convert_one_case, task_list),
                total=len(task_list),
                desc="nii2mha",
                leave=False,
                dynamic_ncols=True):
                pass
    else:
        for args in tqdm(task_list, 
                         leave=False, 
                         dynamic_ncols=True,
                         desc="nii2mha"):
            convert_one_case(args)


def main():
    parser = argparse.ArgumentParser(description="Convert all NIfTI files in a directory to MHA format.")
    parser.add_argument('input_dir', type=str, help="Containing NIfTI files.")
    parser.add_argument('output_dir', type=str, help="Save MHA files.")
    parser.add_argument('--mp', action='store_true', help="Use multiprocessing.")
    parser.add_argument('--spacing', type=float, nargs=3, default=None, help="Resample to this spacing.")
    parser.add_argument('--size', type=int, nargs=3, default=None, help="Crop to this size.")
    args = parser.parse_args()
    
    convert_and_save_nii_to_mha(args.input_dir, args.output_dir, args.mp, args.spacing, args.size)


if __name__ == "__main__":
    main()