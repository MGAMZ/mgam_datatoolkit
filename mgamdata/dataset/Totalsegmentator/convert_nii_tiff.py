import os
import argparse
import multiprocessing
from tqdm import tqdm

import SimpleITK as sitk

from mgamdata.io.nii_toolkit import convert_nii_sitk
from mgamdata.io.sitk_toolkit import merge_masks
from mgamdata.dataset.Totalsegmentator.meta import CLASS_INDEX_MAP




def process_file(args):
    nii_path, output_dir, input_dir = args
    
    # 构建路径，保持文件存储结构不变
    relative_path = os.path.relpath(os.path.dirname(nii_path), input_dir)
    output_path = os.path.join(output_dir, relative_path)
    output_file = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0] + '.mha'
    output_file_path = os.path.join(output_path, output_file)
    os.makedirs(output_path, exist_ok=True)
    if os.path.exists(output_file_path):
        return
    
    # 转换为SimpleITK格式
    sitk_image = convert_nii_sitk(nii_path)
    # 保存为mha格式
    sitk.WriteImage(sitk_image, output_file_path, useCompression=True)



def merge_one_case_segmentations(case_path: str):
    ct_path = os.path.join(case_path, 'ct.mha')
    segmentation_path = os.path.join(case_path, 'segmentations')
    if not os.path.exists(ct_path):
        return
    
    # 融合独立的annotation
    label_itk_image = merge_masks(
        mha_paths=[os.path.join(segmentation_path, file) 
                   for file in os.listdir(segmentation_path)
                   if file.endswith('.mha')],
        class_index_map=CLASS_INDEX_MAP
    )
    sitk.WriteImage(
        label_itk_image,
        os.path.join(case_path, 'segmentations.mha'),
        useCompression=True)



def convert_and_save_all_nii_to_mha(input_dir: str, output_dir: str, use_mp: bool):
    file_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_path = os.path.join(root, file)
                file_list.append((nii_path, output_dir, input_dir))
    
    if use_mp:
        with multiprocessing.Pool() as pool:
            for _ in tqdm(
                pool.imap_unordered(process_file, file_list),
                total=len(file_list),
                desc="nii2mha",
                leave=False,
                dynamic_ncols=True):
                pass
    else:
        for args in tqdm(file_list, 
                         leave=False, 
                         dynamic_ncols=True,
                         desc="nii2mha"):
            process_file(args)



def merge_all_segmentations(input_dir: str, use_mp: bool):
    case_list = [os.path.join(input_dir, case) 
                 for case in os.listdir(input_dir) 
                 if os.path.isdir(os.path.join(input_dir, case))]
    
    if use_mp:
        with multiprocessing.Pool() as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    merge_one_case_segmentations, 
                    case_list), 
                total=len(case_list),
                desc="merging mask",
                leave=False, 
                dynamic_ncols=True):
                
                pass
    
    else:
        for case in tqdm(case_list,
                         desc="merging mask",
                         leave=False,
                         dynamic_ncols=True):
            merge_one_case_segmentations(case)



def main():
    parser = argparse.ArgumentParser(description="Convert all NIfTI files in a directory to MHA format.")
    parser.add_argument('input_dir', type=str, help="Containing NIfTI files.")
    parser.add_argument('output_dir', type=str, help="Save MHA files.")
    parser.add_argument('--mp', action='store_true', help="Use multiprocessing.")
    args = parser.parse_args()
    
    convert_and_save_all_nii_to_mha(args.input_dir, args.output_dir, args.mp)
    merge_all_segmentations(args.output_dir, args.mp)




if __name__ == "__main__":
    main()