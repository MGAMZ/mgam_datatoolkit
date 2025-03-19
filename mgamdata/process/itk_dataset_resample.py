import os
import argparse
import json
from tqdm import tqdm
from collections.abc import Sequence
from multiprocessing import Pool

import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size



def resample_one_sample(args) -> tuple[sitk.Image, sitk.Image|None] | None:
    """
    Resample a single sample image and its corresponding label image.

    Args:
        image_itk_path (str): The file path of the input image.
        label_itk_path (str): The file path of the input label image.
        params (Sequence[float|int]): The target parameters (spacing or size) for resampling.
        out_image_folder (str): The output folder for the resampled image.
        out_label_folder (str): The output folder for the resampled label image.
        use_size (bool): Whether to resample by size instead of spacing.

    Returns:
        A tuple containing the resampled image and label image, or None if the output files already exist.
    """
    image_itk_path, label_itk_path, params, out_image_folder, out_label_folder, use_size = args

    itk_name = os.path.basename(image_itk_path)
    target_image_path = os.path.join(out_image_folder, itk_name)
    target_label_path = os.path.join(out_label_folder, itk_name)
    if os.path.exists(target_image_path) and os.path.exists(target_label_path):
        return None

    image_itk = sitk.ReadImage(image_itk_path)
    label_itk = None
    if os.path.exists(label_itk_path):
        label_itk = sitk.ReadImage(label_itk_path)
    
    # 根据模式选择重采样方法
    if use_size:
        image_resampled = sitk_resample_to_size(image_itk, params, "image")
        if label_itk:
            label_resampled = sitk_resample_to_size(label_itk, params, "label")
    else:
        image_resampled = sitk_resample_to_spacing(image_itk, params, "image")
        if label_itk:
            label_resampled = sitk_resample_to_spacing(label_itk, params, "label")

    target_image_path = target_image_path.replace(".nii.gz", ".mha").replace(".nii", ".mha")
    sitk.WriteImage(image_resampled, target_image_path, useCompression=True)
    if label_itk:
        sitk.WriteImage(label_resampled, target_label_path, useCompression=True)
    return image_resampled, label_resampled if label_itk else None


def resample_standard_dataset(
    source_root: str, 
    params: Sequence[float|int], 
    dest_root: str, 
    mp: bool = False, 
    workers: int|None = None, 
    use_size: bool = False
):
    """
    Resample a standard dataset.
    
    Args:
        source_root (str): The root folder of the source dataset.
        params (Sequence[float|int]): The target parameters (spacing or size) for resampling.
        dest_root (str): The root folder of the destination dataset.
        mp (bool): Whether to use multiprocessing.
        use_size (bool): Whether to resample by size instead of spacing.
    """
    source_image_folder = os.path.join(source_root, "image")
    source_label_folder = os.path.join(source_root, "label")
    dest_image_folder = os.path.join(dest_root, "image")
    dest_label_folder = os.path.join(dest_root, "label")
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_label_folder, exist_ok=True)

    image_itk_paths = [
        os.path.join(source_image_folder, f)
        for f in os.listdir(source_image_folder)
        if f.endswith((".mha", ".nii", ".nii.gz", "mhd"))
    ]
    label_itk_paths = [i.replace("image", "label") 
                       for i in image_itk_paths]

    if mp:
        task_list = [
            (
                image_itk_paths[i],
                label_itk_paths[i],
                params,
                dest_image_folder,
                dest_label_folder,
                use_size,
            )
            for i in range(len(image_itk_paths))
        ]
        
        with (
            Pool(processes=workers) as pool,
            tqdm(
                total=len(image_itk_paths),
                desc="Resampling",
                leave=False,
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
            leave=False,
            dynamic_ncols=True,
        ) as pbar:
            for i in range(len(image_itk_paths)):
                resample_one_sample(
                (
                    image_itk_paths[i],
                    label_itk_paths[i],
                    params,
                    dest_image_folder,
                    dest_label_folder,
                    use_size,
                )
            )
            pbar.update(len(image_itk_paths))


def parse_args():
    parser = argparse.ArgumentParser(description="Resample a standard dataset.")
    parser.add_argument("source_root", type=str, help="The root folder of the source dataset.")
    parser.add_argument("dest_root", type=str, help="The root folder of the destination dataset.")
    parser.add_argument("--mp", action="store_true", help="Whether to use multiprocessing.")
    parser.add_argument("--workers", type=int, default=None, help="The number of workers for multiprocessing.")
    
    # 互斥参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spacing", type=float, nargs="+", default=None, help="The target spacing for resampling.")
    group.add_argument("--size", type=int, nargs="+", default=None, help="The target size for resampling.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.dest_root, exist_ok=True)
    json.dump(vars(args),
              open(os.path.join(args.dest_root, "resample_configs.json"), "w"), 
              indent=4)
    resample_standard_dataset(
        args.source_root,
        args.size or args.spacing,
        args.dest_root,
        args.mp, 
        args.workers,
        use_size=args.size is not None)



if __name__ == '__main__':
    main()
