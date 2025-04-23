import os
import pdb
import argparse
import json
from tqdm import tqdm
from collections.abc import Sequence
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size



def resample_one_sample(args) -> tuple[sitk.Image, sitk.Image|None] | None:
    """
    Resample a single sample image and its corresponding label image based on
    dimension-wise spacing and size rules.

    Args:
        args (tuple): A tuple containing:
            image_itk_path (str): The file path of the input image.
            label_itk_path (str): The file path of the input label image.
            target_spacing (Sequence[float]): Target spacing per dimension (-1 to ignore).
            target_size (Sequence[int]): Target size per dimension (-1 to ignore).
            out_image_folder (str): The output folder for the resampled image.
            out_label_folder (str): The output folder for the resampled label image.

    Returns:
        A tuple containing the resampled image and label image, or None if the output files already exist.
    """
    image_itk_path, label_itk_path, target_spacing, target_size, out_image_folder, out_label_folder = args
    img_dim = 3

    # 路径
    itk_name = os.path.basename(image_itk_path)
    target_image_path = os.path.join(out_image_folder, itk_name)
    target_label_path = os.path.join(out_label_folder, itk_name)
    # 检查目标文件是否已存在（考虑 .mha 后缀）
    potential_target_image_path = target_image_path.replace(".nii.gz", ".mha").replace(".nii", ".mha")
    potential_target_label_path = target_label_path.replace(".nii.gz", ".mha").replace(".nii", ".mha")
    if os.path.exists(potential_target_image_path) and (not os.path.exists(label_itk_path) or os.path.exists(potential_target_label_path)):
         tqdm.write(f"Skipping {itk_name}, output exists.")
         return None

    # 读取
    try:
        image_itk = sitk.ReadImage(image_itk_path)
        label_itk = None
        if os.path.exists(label_itk_path):
            label_itk = sitk.ReadImage(label_itk_path)
        else:
            tqdm.write(f"Warning: Label file not found for {image_itk_path}, skipping label resampling.")
    except Exception as e:
        tqdm.write(f"Error reading {image_itk_path} or {label_itk_path}: {e}")
        return None

    # --- 阶段一：Spacing 重采样 ---
    orig_spacing = image_itk.GetSpacing()[::-1]
    effective_spacing = list(orig_spacing)
    needs_spacing_resample = False
    for i in range(img_dim):
        if target_spacing[i] != -1:
            effective_spacing[i] = target_spacing[i]
            needs_spacing_resample = True

    image_after_spacing = image_itk
    label_after_spacing = label_itk

    if needs_spacing_resample and not np.allclose(effective_spacing, orig_spacing):
        tqdm.write(f"Resampling {itk_name} to spacing {effective_spacing}...")
        image_after_spacing = sitk_resample_to_spacing(image_itk, effective_spacing, "image")
        if label_itk:
            label_after_spacing = sitk_resample_to_spacing(label_itk, effective_spacing, "label")

    # --- 阶段二：Size 重采样 ---
    current_size = image_after_spacing.GetSize()[::-1]
    effective_size = list(current_size)
    needs_size_resample = False
    for i in range(img_dim):
        if target_size[i] != -1:
            effective_size[i] = target_size[i]
            needs_size_resample = True

    image_resampled = image_after_spacing
    label_resampled = label_after_spacing

    if needs_size_resample and effective_size != list(current_size):
        tqdm.write(f"Resampling {itk_name} to size {effective_size}...")
        image_resampled = sitk_resample_to_size(image_after_spacing, effective_size, "image")
        if label_itk and label_after_spacing: # 确保 label 存在且经过了第一阶段
             label_resampled = sitk_resample_to_size(label_after_spacing, effective_size, "label")

    # --- 阶段三：方向重采样 ---
    image_resampled = sitk.DICOMOrient(image_resampled, 'LPI')
    if label_itk and label_resampled:
        label_resampled = sitk.DICOMOrient(label_resampled, 'LPI')

    # 写入
    target_image_path = potential_target_image_path
    target_label_path = potential_target_label_path
    try:
        sitk.WriteImage(image_resampled, target_image_path, useCompression=True)
        if label_itk and label_resampled:
            sitk.WriteImage(label_resampled, target_label_path, useCompression=True)
    except Exception as e:
        tqdm.write(f"Error writing {target_image_path} or {target_label_path}: {e}")
        return None

    return image_resampled, label_resampled if label_itk else None


def resample_standard_dataset(
    source_root: str,
    target_spacing: Sequence[float],
    target_size: Sequence[int],
    dest_root: str,
    mp: bool = False,
    workers: int|None = None,
):
    """
    Resample a standard dataset with dimension-wise spacing/size rules.

    Args:
        source_root (str): The root folder of the source dataset.
        target_spacing (Sequence[float]): Target spacing per dimension (-1 to ignore).
        target_size (Sequence[int]): Target size per dimension (-1 to ignore).
        dest_root (str): The root folder of the destination dataset.
        mp (bool): Whether to use multiprocessing.
        workers (int | None): Number of workers for multiprocessing.
    """
    # 路径定义
    source_image_folder = os.path.join(source_root, "image")
    source_label_folder = os.path.join(source_root, "label")
    dest_image_folder = os.path.join(dest_root, "image")
    dest_label_folder = os.path.join(dest_root, "label")
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_label_folder, exist_ok=True)

    # 任务准备
    image_itk_paths = []
    label_itk_paths = []
    if os.path.exists(source_image_folder):
        image_itk_paths = [
            os.path.join(source_image_folder, f)
            for f in os.listdir(source_image_folder)
            if f.endswith((".mha", ".nii", ".nii.gz", "mhd"))
        ]
        label_itk_paths = [
            os.path.join(source_label_folder, os.path.basename(p))
            for p in image_itk_paths
        ]
    else:
        tqdm.write(f"Warning: Source image folder not found: {source_image_folder}")
        return

    if not image_itk_paths:
        tqdm.write("No image files found to process.")
        return

    task_list = [
        (
            image_itk_paths[i],
            label_itk_paths[i],
            target_spacing,
            target_size,
            dest_image_folder,
            dest_label_folder,
        )
        for i in range(len(image_itk_paths))
    ]

    # 可选多进程执行
    if mp:
        with (
            Pool(processes=workers) as pool,
            tqdm(
                total=len(image_itk_paths),
                desc="Resampling",
                leave=True,
                dynamic_ncols=True,
            ) as pbar,
        ):
            result_fetcher = pool.imap_unordered(
                func=resample_one_sample,
                iterable=task_list,
            )
            for _ in result_fetcher:
                pbar.update()
    else:
        with tqdm(
            total=len(image_itk_paths),
            desc="Resampling",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for task_args in task_list:
                resample_one_sample(task_args)
                pbar.update()


def parse_args():
    parser = argparse.ArgumentParser(description="Resample a standard dataset with dimension-wise spacing/size rules.")
    parser.add_argument("source_root", type=str, help="The root folder of the source dataset.")
    parser.add_argument("dest_root", type=str, help="The root folder of the destination dataset.")
    parser.add_argument("--mp", action="store_true", help="Whether to use multiprocessing.")
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")

    # 允许同时指定，用 -1 表示不指定
    # 类型为 str 先接收，方便处理 -1
    parser.add_argument("--spacing", type=str, nargs='+', default=["-1", "-1", "-1"],
                        help="Target spacing (ZYX order). Use -1 for dimensions to ignore spacing rule. e.g., 1.5 -1 1.5")
    parser.add_argument("--size", type=str, nargs='+', default=["-1", "-1", "-1"],
                        help="Target size (ZYX order). Use -1 for dimensions to ignore size rule. e.g., -1 256 256")

    return parser.parse_args()


def main():
    args = parse_args()
    img_dim = 3 # 假设处理3D图像

    # --- 参数转换和验证 ---
    try:
        # 转换 spacing 为 float, size 为 int
        target_spacing = [float(s) for s in args.spacing]
        target_size = [int(s) for s in args.size]

        # 检查列表长度是否匹配维度
        if len(target_spacing) != img_dim:
            raise ValueError(f"--spacing must have {img_dim} values (received {len(target_spacing)})")
        if len(target_size) != img_dim:
             raise ValueError(f"--size must have {img_dim} values (received {len(target_size)})")

        # 验证每个维度的互斥性
        for i in range(img_dim):
            if target_spacing[i] != -1 and target_size[i] != -1:
                raise ValueError(f"Dimension {i} (ZYX order) cannot have both spacing ({target_spacing[i]}) and size ({target_size[i]}) specified. Use -1 for one of them.")

        # 检查是否至少指定了一个重采样操作
        if all(s == -1 for s in target_spacing) and all(sz == -1 for sz in target_size):
             tqdm.write("Warning: No resampling specified (all spacing and size values are -1).")
             return

    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        return

    os.makedirs(args.dest_root, exist_ok=True)
    print(f"Resampling {args.source_root} to {args.dest_root}")
    print(f"  Target Spacing (ZYX): {target_spacing}")
    print(f"  Target Size (ZYX): {target_size}")

    # 保存配置信息
    config_data = vars(args)
    config_data['target_spacing_validated'] = target_spacing
    config_data['target_size_validated'] = target_size
    try:
        with open(os.path.join(args.dest_root, "resample_configs.json"), "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}")

    # 执行
    resample_standard_dataset(
        args.source_root,
        target_spacing,
        target_size,
        args.dest_root,
        args.mp,
        args.workers,
    )
    print(f"Resampling completed. The resampled dataset is saved in {args.dest_root}.")



if __name__ == '__main__':
    main()