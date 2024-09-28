'''
    给定gt和pred的两组mha文件
    计算其指标
'''
import argparse
import os
import os.path as osp
import pdb
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from typing import List

import pandas as pd
import numpy as np
import SimpleITK as sitk

from ..criterions.segment import evaluation_dice, evaluation_area_metrics, evaluation_hausdorff_distance_3D



GT_FOLDERS = [
    '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Check_8081/mha_original_EngineerSort/label',
    '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Batch6_8016/mha_original_EngineerSort/label',
    '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Batch5_7986/mha_original_EngineerSort/label',
    '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Batch1234/mha_IdentityDevelopSort_AllinOne/label',
]



def calculate_one_pair(gt_path:str, pred_path:str, only_L3:bool=False, invert:bool=False):
    """计算一个mha样本对的dice

    Args:
        gt_path (str): index ground truth [Z, Y, X]
        pred_path (str): prediction [Z, Y, X]
        only_L3 (bool, optional): Clip to L3 according to label valid area.

    Returns:
        _type_: _description_
    """
    if gt_path is None:
        return {
        'seriesUID': Path(pred_path).stem,
        'dice': np.nan,
        'iou': np.nan,
        'recall': np.nan,
        'precision': np.nan
    }
    try:
        gt = sitk.ReadImage(gt_path)
    except Exception as e:
        raise RuntimeError(f"Read mha File({gt_path}) Failed: {e}")
    try:
        pred = sitk.ReadImage(pred_path)
    except Exception as e:
        raise RuntimeError(f"Read mha File({pred_path}) Failed: {e}")
    
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
    gt_data_channel = np.stack([gt_data == i for i in range(1, 5)], axis=0)
    pred_data_channel = np.stack([pred_data == i for i in range(1, 5)], axis=0)
    
    # 逐类计算metric
    dices, ious, recalls, precisions = [], [], [], []
    hausdorff = evaluation_hausdorff_distance_3D(gt_data, pred_data)
    for i in range(4):
        dice = evaluation_dice(gt_data_channel[i], pred_data_channel[i])
        iou, recall, precision = evaluation_area_metrics(
            gt_data_channel[i].astype(np.uint8), pred_data_channel[i].astype(np.uint8))
        
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
        'precision': np.stack(precisions),
        'hausdorff': hausdorff,
    }


# NOTE 输入的所有gt文件夹是有优先级顺序的，只会返回最先找到的gt路径
def search_gt_file(gt_folders:List[str], seriesUID:str):
    for gt_folder in gt_folders:
        for roots, dirs, files in os.walk(gt_folder):
            for file in files:
                if file.rstrip('.mha')==seriesUID and 'label' in roots:
                    return os.path.join(roots, file)
    else:
        print(f"Can't find gt file for {seriesUID}.")



def evaluate_one_folder(pred_folder:str,
                        only_L3:bool=False,
                        use_mp:bool=False,
                        invert:bool=False):
    pred_files = sorted([osp.join(pred_folder, file)
                         for file in os.listdir(pred_folder)
                         if file.endswith('.mha')])
    gt_files = [search_gt_file(GT_FOLDERS, seriesUID) for seriesUID in 
                        [Path(file).stem for file in pred_files]]
    metric_list = []

    if use_mp:
        with mp.Pool(24) as p:
            results = []
            for gt_path, pred_path in zip(gt_files, pred_files):
                result = p.apply_async(
                    calculate_one_pair,
                    args=(gt_path, pred_path, only_L3, invert))
                results.append(result)

            for result in tqdm(results, 
                               desc='Evaulating',
                               total=len(gt_files),
                               dynamic_ncols=True,
                               leave=False):
                metric_list.append(result.get())

    else:
        for gt_path, pred_path in tqdm(zip(gt_files, pred_files),
                                       desc='Evaulating',
                                       total=len(gt_files),
                                       dynamic_ncols=True,
                                       leave=False):
            out = calculate_one_pair(gt_path, pred_path, only_L3, invert)
            metric_list.append(out)

    return metric_list



def parser_args():
    parser = argparse.ArgumentParser('evaluate from mha files.')
    parser.add_argument('pred_root', type=str, help='预测文件夹')
    parser.add_argument('--whole-series', action='store_true', default=False, 
                        help='仅计算L3节段指标')
    parser.add_argument('--invert', action='store_true', default=False, 
                        help='在评估时将其中一者颠倒评估。这是为了避免Z轴排序不一致导致的指标错误。')
    parser.add_argument('--mp', action='store_true', default=False)
    return parser.parse_args()




if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = parser_args()
    
    # 执行评估
    # metric_list中元素的数量是样本数量
    # 每个元素是一个字典，包含四个指标的四个类的值和自身的SeriesUID
    metric_list = evaluate_one_folder(args.pred_root,
                                      only_L3=not args.whole_series,
                                      invert=args.invert,
                                      use_mp=args.mp)
    
    # 汇总整理
    result = pd.DataFrame(metric_list).dropna().drop_duplicates('seriesUID')
    # 将一个metric的四个类分解成四个单独的列
    for metric_column in result.drop(columns='seriesUID').columns:
        colume_names = [f'{metric_column}_腰大肌', 
                        f'{metric_column}_其他骨骼肌', 
                        f'{metric_column}_皮下脂肪', 
                        f'{metric_column}_内脏脂肪']
        result[colume_names] = pd.DataFrame(result[metric_column].tolist(), index=result.index)
        result = result.drop(columns=metric_column)
        # 计算这一metric的四类平均值
        result[f'{metric_column}_Avg'] = result[colume_names].mean(axis=1)
    # 保存csv
    result.to_csv(os.path.join(args.pred_root, 'evaluation.csv'), index=False)
    print(result)
