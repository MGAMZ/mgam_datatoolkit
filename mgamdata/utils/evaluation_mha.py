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

import pandas as pd
import numpy as np
import SimpleITK as sitk

from ..criterions.segment import evaluation_dice, evaluation_area_metrics, evaluation_hausdorff_distance_3D




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



def evaluate_one_folder(gt_folder:str, 
                        pred_folder:str, 
                        only_L3:bool=False, 
                        use_mp:bool=False, 
                        invert:bool=False):
    pred_files = sorted([file for file in os.listdir(pred_folder) if file.endswith('.mha')])
    gt_files = [file.replace(pred_folder, gt_folder) for file in pred_files]
    
    metric_list = []
    
    if use_mp:
        with mp.Pool(24) as p:
            results = []
            for gt_file, pred_file in zip(gt_files, pred_files):
                gt_path = osp.join(gt_folder, gt_file)
                pred_path = osp.join(pred_folder, pred_file)
                result = p.apply_async(calculate_one_pair, 
                                       args=(gt_path, pred_path, only_L3, invert))
                results.append(result)
            
            for result in tqdm(results, 
                               desc='Evaulating',
                               total=len(gt_files),
                               dynamic_ncols=True,
                               leave=False):
                metric_list.append(result.get())
    
    else:
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
    mp.set_start_method('spawn', force=True)
    args = parser_args()
    
    # 执行评估
    # metric_list中元素的数量是样本数量
    # 每个元素是一个字典，包含四个指标的四个类的值和自身的SeriesUID
    metric_list = evaluate_one_folder(args.gt_root, 
                                      args.pred_root, 
                                      only_L3=args.whole_series, 
                                      invert=args.invert,
                                      use_mp=True)
    
    # 汇总整理
    result = pd.DataFrame(metric_list)
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
