import os
import os.path as osp
import pdb
from multiprocessing import Pool
from colorama import Fore, Style
from tqdm import tqdm
from pprint import pprint
from typing import List
from pathlib import Path
from typing_extensions import deprecated

import pandas as pd
import SimpleITK as sitk
import numpy as np

from aitrox.criterions import dice_loss_array, accuracy_array
from aitrox.utils.sitk_toolkit import sitk_resample_to_image


# Inference may contain annotations outside the L3 area.
# Providing having CSV annotations that contain the L3 locolization,
# we can erase all the annotations outside the L3 area.
# This function will also change mask into channel-seperated.
@deprecated('This function contains too much codes, too verbose. Improvement on the way.')
def Evaluation(anno_csv_path:str|None, 
               folder_to_process:str, 
               folder_to_output:str|None=None,
               ground_truth_mha_root:str|None=None,
               loss_calc_method:str='L3'):
    
    # prepare
    if ground_truth_mha_root is not None:
        criterions = {'dice': [], 'acc': []}
    if folder_to_output is not None:
        os.makedirs(folder_to_output, exist_ok=True)
    if anno_csv_path is not None:
        L3_anno = pd.read_csv(anno_csv_path, 
                              header=0, 
                              usecols=['序列编号', 'L3节段起始层数', 'L3节段终止层数'])
    
    # process each sample
    for mha_file in tqdm(os.listdir(folder_to_process),
                         desc='Evaluating'):
        if not mha_file.endswith('.mha'):
            continue
        
        # Load original files
        ori_mha_path = osp.join(folder_to_process, mha_file)
        ori_mha = sitk.ReadImage(ori_mha_path)
        ori_mha_image = sitk.GetArrayFromImage(ori_mha) # [D, H, W]
        
        if ground_truth_mha_root is not None:
            gt_mha_path = osp.join(ground_truth_mha_root, mha_file.replace('_0000', ''))
            if not osp.exists(gt_mha_path):
                continue
            gt_mha = sitk.ReadImage(gt_mha_path)
            gt_array = sitk.GetArrayFromImage(gt_mha)   # [D, H, W]
            inverted_gt_array = np.flip(gt_array, axis=0)
        
        # Read L3 location from csv annotation files.
        if anno_csv_path is not None:
            try:
                L3_start_idx, L3_end_idx = L3_anno.loc[L3_anno['序列编号'] == mha_file.rstrip('.mha'), 
                                                    ['L3节段起始层数', 'L3节段终止层数']
                                                    ].values.flatten()
                L3_start_idx, L3_end_idx = int(L3_start_idx), int(L3_end_idx)
            except:
                continue
        
        # If there are no independent L3 annotations,
        # I fall back to use the gt array to see which slice have valid annotation.
        # Because the annotation should only locate at L3 area.
        elif ground_truth_mha_root is not None:
            valid_slices, = np.where(inverted_gt_array.any(axis=(1, 2)))
            L3_start_idx, L3_end_idx = valid_slices.min(), valid_slices.max()
        else:
            L3_start_idx, L3_end_idx = None, None
        
        # ATTENTION AND WARNING: 
        # The L3 annotation value is inverted and was counted begin with maximum Z value slice.
        inverted_mha_image = np.flip(ori_mha_image, axis=0)
        if L3_start_idx is not None and L3_end_idx is not None:
            inverted_mha_image[:L3_start_idx] = 0
            inverted_mha_image[L3_end_idx:] = 0
        
        # If provided gt, metric calculation is available.
        if ground_truth_mha_root is not None:
            if loss_calc_method == 'L3':
                pred_WithinL3Area = inverted_mha_image[L3_start_idx:L3_end_idx]
                gt_WithinL3Area = inverted_gt_array[L3_start_idx:L3_end_idx]
            elif loss_calc_method == 'all':
                pred_WithinL3Area = inverted_mha_image
                gt_WithinL3Area = inverted_gt_array
            else:
                raise ValueError(f'Invalid loss calculation method: {loss_calc_method}')
        
            # There may exist empty gt
            if gt_WithinL3Area.max() == 4:
                dice = dice_loss_array(pred_WithinL3Area, gt_WithinL3Area)
                acc  = accuracy_array(pred_WithinL3Area, gt_WithinL3Area)
                criterions['dice'].append(dice)
                criterions['acc'].append(acc)
        
        # Save back.
        for class_idx in range(4):
            case_folder = osp.join(folder_to_output, mha_file).rstrip('.mha')
            class_mask = (inverted_mha_image == (class_idx+1)).astype(np.uint8)
            invert_back_to_normal = np.flip(class_mask, axis=0)
            clipped_mha = sitk.GetImageFromArray(invert_back_to_normal, isVector=False)
            clipped_mha.CopyInformation(ori_mha)
            os.makedirs(case_folder, exist_ok=True)
            sitk.WriteImage(clipped_mha, 
                            osp.join(case_folder, mha_file.rstrip('.mha') + f'_{class_idx}.mha'), 
                            useCompression=True)

    if ground_truth_mha_root is not None:
        reduced_criterions = {}
        for criterion_name, value in criterions.items():
            reduced_criterions[criterion_name] = np.mean(value)
    
    pdb.set_trace()



def clip_none_L3_area_mask(image_array:np.ndarray, L3_area:List[int]):
    # NOTE
    # ATTENTION AND WARNING: 
    # The L3 annotation value is inverted and was counted begin with maximum Z value slice.
    inverted_array = np.flip(image_array, axis=0)
    
    inverted_array[:L3_area[0]] = 0
    inverted_array[L3_area[1]:] = 0
    
    normal_array = np.flip(inverted_array, axis=0)
    return normal_array


def save_as_channel_seperated(
    image_array:np.ndarray, target_mha:sitk.Image, patient_name:str, save_root:str):
    # target_mha is where to copy sitk meta data.
    for class_idx in range(4):
        class_mask = (image_array == (class_idx+1)).astype(np.uint8)
        clipped_mha = sitk.GetImageFromArray(class_mask, isVector=False)
        clipped_mha.CopyInformation(target_mha)
        os.makedirs(save_root, exist_ok=True)
        sitk.WriteImage(clipped_mha, 
                        osp.join(save_root, patient_name + f'_{class_idx}.mha'), 
                        useCompression=True)


# NOTE 将会对齐源mha的Spacing
def convert_one_case(src_mask_mha_path, target_image_mha_path, save_root, L3_area:List[int]):
    patient_name = Path(src_mask_mha_path).stem
    
    # Load sitk image
    source_mha_image = sitk.ReadImage(src_mask_mha_path)
    target_mha_image = sitk.ReadImage(target_image_mha_path)

    source_mha_image = sitk_resample_to_image(
        source_mha_image, 
        target_mha_image, 
        interpolator=sitk.sitkNearestNeighbor,
        output_pixel_type=sitk.sitkUInt8)
    
    mask_array = sitk.GetArrayFromImage(source_mha_image) # [D, H, W]
    
    # clear all mask on none-L3 slices
    if isinstance(L3_area[0], int) and isinstance(L3_area[1], int):
        mask_array = clip_none_L3_area_mask(mask_array, L3_area)
    
    # save
    save_as_channel_seperated(
        mask_array, target_mha_image, patient_name, osp.join(save_root, patient_name))



def convert_mask_mha_to_channel_seperated(src_mask_root, target_image_root, save_root, L3_anno_csv):
    # prepare
    os.makedirs(save_root, exist_ok=True)
    if L3_anno_csv is not None:
        L3_anno = pd.read_csv(L3_anno_csv, 
                            header=0, 
                            usecols=['序列编号', 'L3节段起始层数', 'L3节段终止层数'])
    
    with Pool(16) as p:
        results = []
        
        for mha_file in os.listdir(src_mask_root):
            if not mha_file.endswith('.mha'):
                continue
            
            # Load original files
            src_mask_mha_path = osp.join(src_mask_root, mha_file)
            target_image_mha_path = osp.join(target_image_root, mha_file)
            
            # Read L3 location from csv annotation files.
            try:
                L3_start_idx, L3_end_idx = L3_anno.loc[L3_anno['序列编号'] == mha_file.rstrip('.mha'), 
                                                    ['L3节段起始层数', 'L3节段终止层数']
                                                    ].values.flatten()
                L3_start_idx, L3_end_idx = int(L3_start_idx), int(L3_end_idx)
            except:
                print(Fore.RED + f"Failed to find L3 annotation for {mha_file}" + Style.RESET_ALL)
                L3_start_idx, L3_end_idx = None, None
            
            task = p.apply_async(
                convert_one_case, 
                args=(src_mask_mha_path, target_image_mha_path, save_root, [L3_start_idx, L3_end_idx]))
            results.append(task)
        
        for result in tqdm(results, desc='Converting', leave=True, dynamic_ncols=True):
            result.get()


if __name__ == '__main__':
    '''
        src_mask_root: 需要转换的图像
        target_image_root: 用于对齐spacing的图像
        save_root: 保存的根目录
        L3_anno_csv: 包含L3位置信息的csv文件
    '''
    
    convert_mask_mha_to_channel_seperated(
        src_mask_root = "/fileser51/zhangyiqin.sx/Sarcopenia_PreSeg/Data8031/nnUNet_Pred_2D_V2",
        target_image_root = "/fileser51/zhangyiqin.sx/Sarcopenia_PreSeg/Data8031/mha_for_2D_seg_V2",
        save_root = "/fileser51/zhangyiqin.sx/Sarcopenia_PreSeg/Data8031/nnUNet_Pred_2D_V2_ChannelSeperated",
        L3_anno_csv = None,
    )