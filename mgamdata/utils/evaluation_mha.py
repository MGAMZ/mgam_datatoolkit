'''
    给定gt和pred的两组mha文件
    计算其指标
'''
import argparse
import os
import os.path as osp
import pdb
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from pprint import pprint

import torch
import pandas
import numpy as np
import SimpleITK as sitk

from mmseg.models.losses.dice_loss import dice_loss




def calc_dice(gt_data:np.ndarray, pred_data:np.ndarray):
    gt_class = torch.from_numpy(gt_data).cuda()
    pred_class = torch.from_numpy(pred_data).cuda()
    dice = 1 - dice_loss(gt_class[None], pred_class[None], weight=None, ignore_index=None
                            ).cpu().numpy()
    return dice



def calc_area_metric(gt_data:np.ndarray, pred_data:np.ndarray):
    # 计算iou, recall, precision
    gt_class = torch.from_numpy(gt_data).cuda()
    pred_class = torch.from_numpy(pred_data).cuda()
    tp = (gt_class * pred_class).sum()
    fn = gt_class.sum() - tp
    fp = pred_class.sum() - tp
    
    iou = (tp / (tp + fn + fp)).cpu().numpy()
    recall = (tp / (tp + fn)).cpu().numpy()
    precision = (tp / (tp + fp)).cpu().numpy()
    return iou, recall, precision



def calculate_one_pair(gt_path:str, pred_path:str, only_L3:bool=False, invert:bool=False):
    """计算一个mha样本对的dice

    Args:
        gt_path (str): index ground truth [Z, Y, X]
        pred_path (str): prediction [Z, Y, X]
        only_L3 (bool, optional): Clip to L3 according to label valid area.

    Returns:
        _type_: _description_
    """

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
    
    # 逐类计算metric
    dices, ious, recalls, precisions = [], [], [], []
    for i in range(4):
        dice = calc_dice(gt_data[i], pred_data[i])
        iou, recall, precision = calc_area_metric(
            gt_data[i].astype(np.uint8), pred_data[i].astype(np.uint8))
        
        dices.append(dice)
        ious.append(iou)
        recalls.append(recall)
        precisions.append(precision)
    
    # 每个指标包含四个值，代表四个类
    return {
        'seriesUID': Path(pred_path).stem,
        'dice': np.stack(dices),
        'iou': np.stack(ious),
        'recall': np.stack(recalls),
        'precision': np.stack(precisions)
    }



def evaluate_one_folder(gt_folder:str, 
                        pred_folder:str, 
                        only_L3:bool=False, 
                        use_mp:bool=False, 
                        invert:bool=False):
    pred_files = sorted([file for file in os.listdir(pred_folder) if file.endswith('.mha')])
    gt_files = [file.replace(pred_folder, gt_folder) for file in pred_files]
    
    metric_list = []
    
    for gt_file, pred_file in tqdm(zip(gt_files, pred_files),
                                    desc='Evaulating',
                                    total=len(gt_files),
                                    dynamic_ncols=True,
                                    leave=False):
        gt_path = osp.join(gt_folder, gt_file)
        pred_path = osp.join(pred_folder, pred_file)
        out = calculate_one_pair(gt_path, pred_path, only_L3, invert)
        metric_list.append(out)
    
    return metric_list



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
    metric_list = evaluate_one_folder(args.gt_root, args.pred_root, only_L3=args.whole_series, invert=args.invert)
    # metric_list中元素的数量是样本数量
    # 每个元素是一个字典，包含四个指标的四个类的值和自身的SeriesUID
    result = pandas.DataFrame(metric_list)
