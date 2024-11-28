import os
import argparse
from pathlib import Path
import re
import pdb
import multiprocessing
from tqdm import tqdm
from collections.abc import Sequence

import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing_v2, sitk_resample_to_size


def convert_one_case(args):
    img_path, label_path, series_output_folder, spacing, size = args
    output_image_folder = os.path.join(series_output_folder, 'image')
    output_label_folder = os.path.join(series_output_folder, 'label')
    # 构建路径，保持文件存储结构不变
    series_id = Path(img_path).stem
    output_image_mha_path = os.path.join(output_image_folder, f"{series_id}.mha")
    output_label_mha_path = os.path.join(output_label_folder, f"{series_id}.mha")
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    if os.path.exists(output_image_mha_path) and os.path.exists(output_label_mha_path):
        return
    
    # 原始扫描转换为SimpleITK格式并保存
    # 类分离的标注文件合并后保存
    input_image_mha = sitk.ReadImage(img_path)
    input_label_mha = sitk.ReadImage(label_path)
    
    if spacing is not None:
        assert size is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_spacing_v2(input_image_mha, spacing, 'image')
        input_label_mha = sitk_resample_to_spacing_v2(input_label_mha, spacing, 'label')
    if size is not None:
        assert spacing is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_size(input_image_mha, size, 'image')
        input_label_mha = sitk_resample_to_size(input_label_mha, size, 'label')
    
    sitk.WriteImage(input_image_mha, output_image_mha_path, useCompression=True)
    sitk.WriteImage(input_label_mha, output_label_mha_path, useCompression=True)


def convert_and_save_to_mha(input_dir: str, 
                            output_dir: str, 
                            use_mp: bool, 
                            spacing:Sequence[float|int]|None=None, 
                            size:Sequence[float|int]|None=None):
    labels_folder = os.path.join(input_dir, 'seg-lungs-LUNA16')
    labels_list = [file
                   for file in os.listdir(labels_folder) 
                   if file.endswith('.mhd')]
    subset_image_folder_list = ["subset{}".format(i) for i in range(10)]
    images_list = [os.path.join(input_dir, subset_image_folder, file)
                   for subset_image_folder in subset_image_folder_list
                   for file in os.listdir(os.path.join(input_dir, subset_image_folder))
                   if file.endswith('.mhd')]
    
    task_list = []
    for image_path in images_list:
        series_file_name = os.path.basename(image_path)
        if series_file_name in labels_list:
            label_path = os.path.join(labels_folder, series_file_name)
            task_list.append((image_path, label_path, output_dir, spacing, size))
    
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
    
    convert_and_save_to_mha(args.input_dir, args.output_dir, args.mp, args.spacing, args.size)


if __name__ == "__main__":
    main()