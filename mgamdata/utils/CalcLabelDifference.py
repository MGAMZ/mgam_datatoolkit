import argparse
from email.mime import image
import os
import os.path as osp
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk




def generate_mha_difference_map(mha1:str, mha2:str, output_path:str):
    """
    Generate the difference map of two mha files.
    
    Args:
        mha1: The first mha file path.
        mha2: The second mha file path.
        output: The output mha file.
    """
    assert mha1.endswith('.mha') and mha2.endswith('.mha') and output_path.endswith('.mha')
    
    image1 = sitk.ReadImage(mha1)
    image2 = sitk.ReadImage(mha2)
    # assert consistent meta info
    assert image1.GetSpacing() == image2.GetSpacing()
    assert image1.GetOrigin() == image2.GetOrigin()
    assert image1.GetDirection() == image2.GetDirection()
    assert image1.GetSize() == image2.GetSize()
    
    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)
    image1_unique = np.unique(image1_array)
    image2_unique = np.unique(image2_array)
    all_unique_value = np.unique(np.concatenate((image1_unique, image2_unique)))
    
    for value in all_unique_value:
        one_hot_source_1 = image1_array == value
        one_hot_source_2 = image2_array == value
        one_hot_map = np.zeros_like(image1_array)
        # 重叠且不是背景
        one_hot_map[(one_hot_source_1 == one_hot_source_2) * one_hot_source_1 * one_hot_source_2] = 1
        # 仅mha1存在
        one_hot_map[(one_hot_source_1 != one_hot_source_2) * one_hot_source_1] = 2
        # 仅mha2存在
        one_hot_map[(one_hot_source_1 != one_hot_source_2) * one_hot_source_2] = 3
        save_path = output_path.replace('.mha', f'_{int(value)}.mha')
        one_hot_itk = sitk.GetImageFromArray(one_hot_map, isVector=False)
        one_hot_itk.CopyInformation(image1)
        sitk.WriteImage(one_hot_itk, save_path, True)


def process_a_pair_folder(mha_folder1:str, mha_folder2:str, output_folder:str):
    """
    Process a pair of mha folders.
    
    Args:
        mha_folder1: The first mha folder path.
        mha_folder2: The second mha folder path.
    """
    
    mha_files1 = [osp.join(mha_folder1, file) for file in os.listdir(mha_folder1) if file.endswith('.mha')]
    mha_files2 = [osp.join(mha_folder2, file) for file in os.listdir(mha_folder2) if file.endswith('.mha')]
    assert len(mha_files1) == len(mha_files2)
    
    for mha1, mha2 in tqdm(zip(mha_files1, mha_files2), 
                           desc="Processing", 
                           total=len(mha_files1), 
                           dynamic_ncols=True, 
                           leave=False):
        assert osp.basename(mha1) == osp.basename(mha2)
        output_path = osp.join(output_folder, osp.basename(mha1))
        generate_mha_difference_map(mha1, mha2, output_path)


def parse_args():
    parser = argparse.ArgumentParser("Generate the difference map of two mha folders.")
    parser.add_argument("mha_folder1", type=str, help="The first mha folder path, which will be subtracted.")
    parser.add_argument("mha_folder2", type=str, help="The second mha folder path, which will to be the subtractor.")
    parser.add_argument("output_folder", type=str, help="The output folder.")
    return parser.parse_args()

if __name__ == '__main__':
    """ Label Definition
    0: Empty
    1: Overlap
    2: only mha1
    3: only mha2
    """
    
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    process_a_pair_folder(args.mha_folder1, args.mha_folder2, args.output_folder)
