import os
import argparse
import glob
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import (
    sitk_resample_to_spacing, 
    sitk_resample_to_size, 
    sitk_resample_to_image)

BraTs2024_MODALITIES = ["t1c", "t1n", "t2f", "t2w"]



def convert_case(case_dir, dest_root, spacing=None, size=None):
    """
    转换单个 Case 的图像和标签。

    Args:
        case_dir (str): Case 目录。
        dest_root (str): 目标数据根目录。
        spacing (tuple, optional): 重采样到此像素间距. Defaults to None.
        size (tuple, optional): 重采样到此大小. Defaults to None.
    """
    case_name = os.path.basename(case_dir)
    phase = os.path.basename(os.path.dirname(case_dir))

    # 读取图像和标签
    images = {}
    label_path = None
    for file in glob.glob(os.path.join(case_dir, '*.nii.gz')):
        file_name = os.path.basename(file)
        if 'seg' in file_name:
            label_path = file
        else:
            for mod in BraTs2024_MODALITIES:
                if mod in file_name:
                    images[mod] = file
                    break

    # 读取标签 (只读取一次)
    label = None
    if label_path:
        try:
            label = sitk.ReadImage(label_path)
        except Exception as e:
            print(f"Failed to read label: {label_path} - {e}")
            return

    # 获取第一个图像，并进行基于spacing或size的resample
    first_modality = next(iter(images))
    first_image_path = images[first_modality]
    try:
        first_image = sitk.ReadImage(first_image_path)

        # Resample 第一个图像
        if spacing is not None:
            first_image = sitk_resample_to_spacing(first_image, spacing, "image")
            if not isinstance(first_image, sitk.Image):
                print(f"Resample to spacing failed for {first_image_path}: {first_image}")
                return
        elif size is not None:
            first_image = sitk_resample_to_size(first_image, size, "image")
            if not isinstance(first_image, sitk.Image):
                print(f"Resample to size failed for {first_image_path}: {first_image}")
                return
    except Exception as e:
        print(f"Failed to read first image: {first_image_path} - {e}")
        return

    # 对齐label到第一个图像
    if label is not None and label_path is not None:
        label = sitk_resample_to_image(label, first_image, "label")
        if not isinstance(label, sitk.Image):
            print(f"Resample label to image failed for {label_path}: {label}")
            label = None
        output_label_path = os.path.join(dest_root, phase, case_name, 'label.mha')
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        sitk.WriteImage(label, output_label_path, useCompression=True)

    # 转换其他图像，并对齐到第一个图像
    for modality, image_path in images.items():
        # 跳过第一个图像，因为它已经被处理
        if modality == first_modality:
            output_image_path = os.path.join(dest_root, phase, case_name, modality + '.mha')
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            sitk.WriteImage(first_image, output_image_path, useCompression=True)
            continue

        try:
            image = sitk.ReadImage(image_path)

            # 对齐到第一个图像
            image = sitk_resample_to_image(image, first_image, "image")
            if not isinstance(image, sitk.Image):
                continue

            output_image_path = os.path.join(dest_root, phase, case_name, modality + '.mha')
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            sitk.WriteImage(image, output_image_path, useCompression=True)

        except Exception as e:
            pass


def convert_brats_to_mha(input_dir, dest_root, spacing=None, size=None, use_mp=False):
    """
    将 BraTs2024 数据集中的 nii.gz 文件转换为 mha 格式，并按照指定的目录结构进行存储。

    Args:
        input_dir (str): 原始数据根目录。
        dest_root (str): 目标数据根目录。
        spacing (tuple, optional): 重采样到此像素间距. Defaults to None.
        size (tuple, optional): 重采样到此大小. Defaults to None.
        use_mp (bool, optional): 是否使用多进程. Defaults to False.
    """

    # 不需要预先创建模态文件夹，convert_case函数会负责创建

    # 收集所有 case 目录
    case_dirs = []
    for phase in ['train', 'val']:
        phase_dir = os.path.join(input_dir, phase)
        case_dirs.extend(glob.glob(os.path.join(phase_dir, '*')))  # 匹配所有文件夹

    # 多进程处理
    partial_convert_func = partial(convert_case, dest_root=dest_root, spacing=spacing, size=size)
    
    if use_mp:
        with mp.Pool(mp.cpu_count()) as pool:
            with tqdm(
                total=len(case_dirs), 
                desc="Converting BraTs2024", 
                dynamic_ncols=True, 
                leave=False
            ) as pbar:
                for _ in pool.imap_unordered(partial_convert_func, case_dirs):
                    pbar.update()
    else:
        with tqdm(total=len(case_dirs), desc="Converting Cases") as pbar:
            for case_dir in case_dirs:
                partial_convert_func(case_dir)
                pbar.update()


def parse_args():
    parser = argparse.ArgumentParser(description="Convert all NIfTI files in a directory to MHA format.")
    parser.add_argument("input_dir", type=str, help="Containing NIfTI files.")
    parser.add_argument("output_dir", type=str, help="Save MHA files.")
    parser.add_argument("--mp", action="store_true", help="Use multiprocessing.")
    parser.add_argument("--spacing", type=float, nargs=3, default=None, help="Resample to this spacing.")
    parser.add_argument("--size", type=int, nargs=3, default=None, help="Crop to this size.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    input_dir = args.input_dir
    dest_root = args.output_dir
    spacing = tuple(args.spacing) if args.spacing else None
    size = tuple(args.size) if args.size else None
    mp_enabled = args.mp

    convert_brats_to_mha(input_dir, dest_root, spacing, size, mp_enabled)