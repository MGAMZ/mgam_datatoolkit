import os
import argparse
from pprint import pprint

import SimpleITK as sitk
from numpy import require

from mgamdata.io.sitk_toolkit import (
    LoadDcmAsSitkImage_JianYingOrder, sitk_resample_to_spacing_v2,
    sitk_resample_to_size)


def find_dcm_sequences(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        dcm_files = [os.path.basename(os.path.join(dirpath, f)) 
                     for f in filenames 
                     if f.lower().endswith('.dcm')]
        if dcm_files:
            yield (os.path.basename(dirpath), sorted(dcm_files))


def convert_one_case(args):
    series_input_folder, series_output_folder, spacing, size = args
    # 构建路径，保持文件存储结构不变
    os.makedirs(series_output_folder, exist_ok=True)
    if os.path.exists(series_output_folder):
        return
    
    # 原始扫描转换为SimpleITK格式并保存
    # 类分离的标注文件合并后保存
    input_image_mha, _, _, _ = LoadDcmAsSitkImage_JianYingOrder(series_input_folder, None)
    
    if spacing is not None:
        assert size is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_spacing_v2(input_image_mha, spacing, 'image')
    if size is not None:
        assert spacing is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_size(input_image_mha, size, 'image')
    
    output_image_mha_path = os.path.join(series_output_folder, os.path.basename(series_input_folder).replace('.dcm', '.mha'))
    sitk.WriteImage(input_image_mha, output_image_mha_path, useCompression=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert dcm to mha, TCGA-LUAD - adenocarcinoma of lung")
    parser.add_argument("root", type=str, required=True, help="SeriesUID root walk root.")
    parser.add_argument("dest", type=str, required=True, help="Mha save path.")
    parser.add_argument("--mp", action="store_true", default=False, help="Multiprocessing.")
    parser.add_argument('--spacing', type=float, nargs=3, default=None, help="Resample to this spacing.")
    parser.add_argument('--size', type=int, nargs=3, default=None, help="Crop to this size.")
    return parser.parse_args()


# 示例用法
if __name__ == "__main__":
    root_folder = "/file1/mgam_datasets/TCGA-LUAD"
    for seriesUID, dcm_sequence in find_dcm_sequences(root_folder):
        print(seriesUID, dcm_sequence)





