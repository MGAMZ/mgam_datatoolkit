'''
    给定gt和pred的两组mha文件
    计算其指标
'''
import argparse
import os
import os.path as osp
import pdb
from multiprocessing import Pool
from tqdm import tqdm
from pprint import pprint

import torch
import numpy as np
import SimpleITK as sitk

from mmseg.models.losses.dice_loss import dice_loss




def calculate_one_pair(gt_path:str, pred_path:str, only_L3:bool=False, invert:bool=False):
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
        pred_data = sitk.GetArrayFromImage(pred)
        if invert:
            pred_data = pred_data[::-1].copy()
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
            dice = 1 - dice_loss(gt_class[None], pred_class[None], weight=None, ignore_index=None
                                 ).cpu().numpy()
            dices.append(dice)
        
        return np.stack(dices)
    
    except Exception as e:
        return f"Failed gt: {gt_path} | pred: {pred_path} | reason: {e}"



def evaluate_one_folder(gt_folder:str, pred_folder:str, only_L3:bool=False, use_mp:bool=False, invert:bool=False):
    pred_files = sorted([file for file in os.listdir(pred_folder) if file.endswith('.mha')])
    gt_files = [file.replace(pred_folder, gt_folder) for file in pred_files]
    
    dice_list = []
    failed_list = []
    if use_mp:
        with Pool(32) as p:
            results = []
            for pred_file in tqdm(pred_files,
                                  desc='Evaulating',
                                  dynamic_ncols=True,
                                  leave=False):
                gt_path = osp.join(gt_folder, pred_file)
                pred_path = osp.join(pred_folder, pred_file)
                one_task = p.apply_async(calculate_one_pair, (gt_path, pred_path, only_L3, invert))
                results.append(one_task)
            
            for result in results:
                out = result.get()
                if isinstance(out, np.ndarray):
                    dice_list.append(out)
                else:
                    failed_list.append(out)
    else:
        for gt_file, pred_file in tqdm(zip(gt_files, pred_files),
                                        desc='Evaulating',
                                        total=len(gt_files),
                                        dynamic_ncols=True,
                                        leave=False):
            gt_path = osp.join(gt_folder, gt_file)
            pred_path = osp.join(pred_folder, pred_file)
            out = calculate_one_pair(gt_path, pred_path, only_L3, invert)
            if isinstance(out, np.ndarray):
                dice_list.append(out)
            else:
                failed_list.append(out)
    
    return dice_list, failed_list



def parser_args():
    parser = argparse.ArgumentParser('evaluate from mha files.')
    parser.add_argument('gt_root', type=str, help='GT文件夹')
    parser.add_argument('pred_root', type=str, help='预测文件夹')
    parser.add_argument('--whole-series', action='store_false', default=True, 
                        help='仅计算L3节段指标')
    parser.add_argument('--invert', action='store_true', default=False, 
                        help='在评估时将其中一者颠倒评估。这是为了避免Z轴排序不一致导致的指标错误。')
    return parser.parse_args()




if __name__ == '__main__':
    args = parser_args()
    dice_list, failed_list = evaluate_one_folder(args.gt_root, args.pred_root, only_L3=args.whole_series, invert=args.invert)
    mean_dice = np.mean(dice_list, axis=0)
    pprint(mean_dice)
    pprint(failed_list)
