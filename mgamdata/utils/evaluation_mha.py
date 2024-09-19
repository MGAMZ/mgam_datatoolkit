'''
    给定gt和pred的两组mha文件
    计算其指标
'''
import os
import os.path as osp
import pdb
from multiprocessing import Pool
from tqdm import tqdm

import torch
import numpy as np
import SimpleITK as sitk

from mmdet.models.losses.dice_loss import dice_loss
from aitrox.criterions.segment import dice_loss_tensor




def calculate_one_pair(gt_path:str, pred_path:str, only_L3:bool=False):
    """计算一个mha样本对的dice

    Args:
        gt_path (str): index ground truth [Z, Y, X]
        pred_path (str): prediction [Z, Y, X]
        only_L3 (bool, optional): Clip to L3 according to label valid area.

    Returns:
        _type_: _description_
    """
    
    try:
        gt = sitk.ReadImage(gt_path)
        pred = sitk.ReadImage(pred_path)
        
        gt_data = sitk.GetArrayFromImage(gt)
        pred_data = sitk.GetArrayFromImage(pred)[::-1]
        if gt_data.shape != pred_data.shape:
            return gt_data.shape, pred_data.shape, gt_path, pred_path
        
        # 如果只计算L3的dice, 则在[D, H, W]的D维度上
        # 寻找gt的any有效slice
        # 并相应抛弃其他所有slices
        if only_L3:
            valid_slices = gt_data.any(axis=(1, 2))
            gt_data = gt_data[valid_slices][3:]
            pred_data = pred_data[valid_slices][3:]
        
        # 生成one-hot编码对
        gt_data = np.stack([gt_data == i for i in range(1, 5)], axis=0)
        pred_data = np.stack([pred_data == i for i in range(1, 5)], axis=0)
        
        # 逐类计算dice
        dices = []
        for i in range(4):
            gt_class = torch.from_numpy(gt_data[i]).cuda()
            pred_class = torch.from_numpy(pred_data[i]).cuda()
            dice = 1 - dice_loss(gt_class[None], pred_class[None]).cpu().numpy()
            dices.append(dice)
        
        return np.stack(dices)
    
    except Exception as e:
        return f"Failed {gt_path}, {pred_path}, {e}"


def evaluate_one_folder(gt_folder:str, pred_folder:str, only_L3:bool=False, use_mp:bool=False):
    pred_files = sorted([file for file in os.listdir(pred_folder) if file.endswith('.mha')])
    gt_files = [file.replace(pred_folder, gt_folder) for file in pred_files]
    
    dice_list = []
    if use_mp:
        with Pool(32) as p:
            results = []
            for pred_file in tqdm(pred_files,
                                  desc='Evaulating',
                                  dynamic_ncols=True,
                                  leave=False):
                gt_path = osp.join(gt_folder, pred_file)
                pred_path = osp.join(pred_folder, pred_file)
                one_task = p.apply_async(calculate_one_pair, (gt_path, pred_path, only_L3))
                results.append(one_task)
            
            for result in results:
                dice = result.get()
                if dice:
                    dice_list.append(dice)
    else:
        for gt_file, pred_file in tqdm(zip(gt_files, pred_files),
                                        desc='Evaulating',
                                        total=len(gt_files),
                                        dynamic_ncols=True,
                                        leave=False):
            gt_path = osp.join(gt_folder, gt_file)
            pred_path = osp.join(pred_folder, pred_file)
            dice = calculate_one_pair(gt_path, pred_path, only_L3)
            if isinstance(dice, np.ndarray):
                dice_list.append(dice)
            else:
                print(f"Failed {gt_file}, get {dice}")
    
    return dice_list


if __name__ == '__main__':
    gt_root = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Test_7986/mask_index'
    pred_root = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Test_7986/mm_pred_0.11.0'
    
    dices = evaluate_one_folder(gt_root, pred_root, only_L3=True)
    mean_dice = np.mean(dices, axis=0)
    
    print(mean_dice)
