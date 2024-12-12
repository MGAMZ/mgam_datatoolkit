import os
import argparse
from tqdm import tqdm
from collections.abc import Sequence
from multiprocessing import Pool

import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing_v2


def resample_one_sample(args) -> tuple[sitk.Image, sitk.Image] | None:
    """
    Resample a single sample image and its corresponding label image to a specified spacing.

    Args:
        image_itk_path (str): The file path of the input image.
        label_itk_path (str): The file path of the input label image.
        spacing (Sequence[float]): The target spacing for resampling.
        out_image_folder (str): The output folder for the resampled image.
        out_label_folder (str): The output folder for the resampled label image.

    Returns:
        A tuple containing the resampled image and label image, or None if the output files already exist.
    """
    image_itk_path, label_itk_path, spacing, out_image_folder, out_label_folder = args

    itk_name = os.path.basename(image_itk_path)
    target_image_path = os.path.join(out_image_folder, itk_name)
    target_label_path = os.path.join(out_label_folder, itk_name)
    if os.path.exists(target_image_path) and os.path.exists(target_label_path):
        return None

    image_itk = sitk.ReadImage(image_itk_path)
    label_itk = sitk.ReadImage(label_itk_path)
    image_resampled = sitk_resample_to_spacing_v2(image_itk, spacing, "image")
    label_resampled = sitk_resample_to_spacing_v2(label_itk, spacing, "label")

    sitk.WriteImage(image_resampled, target_image_path, useCompression=True)
    sitk.WriteImage(label_resampled, target_label_path, useCompression=True)
    return image_resampled, label_resampled


def resample_standard_dataset(
    source_root: str, spacing: Sequence[float], dest_root: str, mp: bool = False
):
    source_image_folder = os.path.join(source_root, "image")
    source_label_folder = os.path.join(source_root, "label")
    dest_image_folder = os.path.join(dest_root, "image")
    dest_label_folder = os.path.join(dest_root, "label")
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_label_folder, exist_ok=True)

    image_itk_paths = [
        os.path.join(source_image_folder, f)
        for f in os.listdir(source_image_folder)
        if f.endswith(".mha")
    ]
    label_itk_paths = [
        os.path.join(source_label_folder, f)
        for f in os.listdir(source_label_folder)
        if f.endswith(".mha")
    ]

    if mp:
        task_list = [
            (
                image_itk_paths[i],
                label_itk_paths[i],
                spacing,
                dest_image_folder,
                dest_label_folder,
            )
            for i in range(len(image_itk_paths))
        ]
        
        with (
            Pool() as pool,
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
            res = [
                resample_one_sample(
                    (
                        image_itk_paths[i],
                        label_itk_paths[i],
                        spacing,
                        dest_image_folder,
                        dest_label_folder,
                    )
                )
                for i in range(len(image_itk_paths))
            ]
            pbar.update(len(image_itk_paths))
    return res


def parse_args():
    parser = argparse.ArgumentParser(description="Resample a standard dataset.")
    parser.add_argument("source_root", type=str, help="The root folder of the source dataset.")
    parser.add_argument("dest_root", type=str, help="The root folder of the destination dataset.")
    parser.add_argument("spacing", type=float, nargs="+", help="The target spacing for resampling.")
    parser.add_argument("--mp", action="store_true", help="Whether to use multiprocessing.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    resample_standard_dataset(args.source_root, args.spacing, args.dest_root, args.mp)