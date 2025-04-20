import os
import argparse
import multiprocessing
from typing_extensions import Sequence
from tqdm import tqdm
import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size, nii_to_sitk, merge_masks
from mgamdata.dataset.Totalsegmentator.meta import CLASS_INDEX_MAP



def merge_one_case_segmentations(corresponding_itk_image:sitk.Image, 
                                 case_path: str):
    segmentation_path = os.path.join(case_path, 'segmentations')
    idx_sorted_paths = sorted(
        [os.path.join(segmentation_path, file)
         for file in os.listdir(segmentation_path)
         if file.endswith('.nii.gz')],
        key=lambda x: CLASS_INDEX_MAP[os.path.basename(x)[:-7]]
    )
    merged_itk = merge_masks(idx_sorted_paths)
    merged_itk.CopyInformation(corresponding_itk_image)
    return merged_itk


def convert_one_case(args):
    series_input_folder, series_output_folder, spacing, size = args
    sample_id = os.path.basename(series_input_folder)
    input_image_nii_path = os.path.join(series_input_folder, 'ct.nii.gz')
    output_image_mha_path = os.path.join(series_output_folder, 'image', f'{sample_id}.mha')
    output_anno_mha_path = os.path.join(series_output_folder, 'label', f'{sample_id}.mha')
    os.makedirs(os.path.join(series_output_folder, 'image'), exist_ok=True)
    os.makedirs(os.path.join(series_output_folder, 'label'), exist_ok=True)
    if os.path.exists(output_image_mha_path) and os.path.exists(output_anno_mha_path):
        return
    
    # 原始扫描转换为SimpleITK格式并保存
    # 类分离的标注文件合并后保存
    input_image_mha = nii_to_sitk(input_image_nii_path, "image")
    merged_itk = merge_one_case_segmentations(input_image_mha, series_input_folder)
    
    if spacing is not None:
        assert size is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_spacing(input_image_mha, spacing, 'image')
        merged_itk = sitk_resample_to_spacing(merged_itk, spacing, 'label')
    if size is not None:
        assert spacing is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_size(input_image_mha, size, 'image')
        merged_itk = sitk_resample_to_size(merged_itk, size, 'label')
    
    input_image_mha = sitk.DICOMOrient(input_image_mha, 'LPI')
    merged_itk = sitk.DICOMOrient(merged_itk, 'LPI')
    sitk.WriteImage(input_image_mha, output_image_mha_path, useCompression=True)
    sitk.WriteImage(merged_itk, output_anno_mha_path, useCompression=True)


def convert_and_save_nii_to_mha(input_dir:str,
                                output_dir:str,
                                use_mp:bool,
                                workers:int|None=None,
                                spacing:Sequence[float|int]|None=None,
                                size:Sequence[float|int]|None=None):
    task_list = []
    for series_name in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, series_name)):
            series_input_folder = os.path.join(input_dir, series_name)
            task_list.append((series_input_folder, output_dir, spacing, size))
    
    if use_mp:
        with multiprocessing.Pool(workers) as pool:
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
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help="Number of workers.")
    parser.add_argument('--spacing', type=float, nargs=3, default=None, help="Resample to this spacing.")
    parser.add_argument('--size', type=int, nargs=3, default=None, help="Crop to this size.")
    args = parser.parse_args()
    
    convert_and_save_nii_to_mha(args.input_dir, args.output_dir, args.mp, args.workers, args.spacing, args.size)



if __name__ == "__main__":
    main()