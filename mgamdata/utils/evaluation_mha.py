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
from typing import Union, Dict, Optional

import pandas as pd
import numpy as np
import SimpleITK as sitk

from mgamdata.criterions.segment import (evaluation_dice,
                                         evaluation_area_metrics,
                                         evaluation_hausdorff_distance_3D)
from mgamdata.dataset.RenJi_Sarcopenia.meta import (
    GT_FOLDERS_PRIORITY_ORIGINAL_ENGINEERSORT, CLASS_MAP, CLASS_MAP_AFTER_KMEANS,
    L3_XLSX_PATH)
from mgamdata.dataset.RenJi_Sarcopenia.L3 import find_L3_slices
from mgamdata.utils.search_tool import search_mha_file




def calculate_one_pair_base_segment(voxel_volume, gt_data, pred_data):
    num_classes = len(CLASS_MAP)
    # 生成one-hot编码对 [Class, D, H, W]
    gt_data_channel = np.stack([gt_data == i for i in range(1, num_classes)])
    pred_data_channel = np.stack([pred_data == i for i in range(1, num_classes)])
    
    # 计算hausdorff距离
    hausdorff = evaluation_hausdorff_distance_3D(gt_data_channel, pred_data_channel)
    
    # 逐类计算metric
    dices, ious, recalls, precisions, gt_volumes, pred_volumes = [], [], [], [], [], []
    for i in range(num_classes-1):
        dice = evaluation_dice(gt_data_channel[i], pred_data_channel[i])
        iou, recall, precision = evaluation_area_metrics(
            gt_data_channel[i].astype(np.uint8), pred_data_channel[i].astype(np.uint8))
        gt_volume = gt_data_channel[i].sum() * voxel_volume
        pred_volume = pred_data_channel[i].sum() * voxel_volume
        
        dices.append(dice)
        ious.append(iou)
        recalls.append(recall)
        precisions.append(precision)
        gt_volumes.append(gt_volume)
        pred_volumes.append(pred_volume)
    
    # 每个指标包含四个值，代表四个类
    return {
        'dice': np.stack(dices),
        'iou': np.stack(ious),
        'recall': np.stack(recalls),
        'precision': np.stack(precisions),
        'hausdorff': hausdorff,
        'gt_volume': np.stack(gt_volumes),
        'pred_volume': np.stack(pred_volumes),
    }



def calculate_one_pair_kmeans_post_segment(voxel_volume, pred_data) -> Dict:
    num_classes = len(CLASS_MAP_AFTER_KMEANS)
    # 生成one-hot编码对 [Class, D, H, W]
    pred_data_channel = np.stack([pred_data == i for i in range(1, num_classes)])
    
    # 逐类计算metric
    pred_volumes = []
    for i in range(num_classes-1):
        pred_volume = pred_data_channel[i].sum() * voxel_volume
        pred_volumes.append(pred_volume)
    
    # 每个指标包含四个值，代表四个类
    return {
        'pred_volume': np.stack(pred_volumes),
    }



def calculate_one_pair(gt_path:Union[str, None],
                       pred_path:str,
                       only_L3:bool=False,
                       invert:bool=False,
                       kmeans:bool=False,
                       L3_slices:Optional[Union[str, np.ndarray[int, int]]]=None,
                    ):
    """计算一个mha样本对的dice

    Args:
        gt_path (str): index ground truth [Z, Y, X]
        pred_path (str): prediction [Z, Y, X]
        only_L3 (bool, optional): Clip to L3 according to label valid area.

    Returns:
        _type_: _description_
    """
    if gt_path is None:
        return None
    try:
        gt = sitk.ReadImage(gt_path)
    except Exception as e:
        raise RuntimeError(f"Read mha File({gt_path}) Failed: {e}")
    try:
        pred = sitk.ReadImage(pred_path)
    except Exception as e:
        raise RuntimeError(f"Read mha File({pred_path}) Failed: {e}")
    
    voxel_volume = np.prod(gt.GetSpacing())
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
        if L3_slices is None:
            L3_slices = gt_data.any(axis=(1, 2))
        else:
            L3_slices = (len(gt_data) - L3_slices)[::-1]
        gt_data = gt_data[L3_slices[0]:L3_slices[1]][:-3]
        pred_data = pred_data[L3_slices[0]:L3_slices[1]][:-3]
        
    if kmeans is True:
        metric = calculate_one_pair_kmeans_post_segment(voxel_volume, pred_data)
    else:
        metric = calculate_one_pair_base_segment(voxel_volume, gt_data, pred_data)
        
    metric['seriesUID'] = Path(pred_path).stem
    return metric



def evaluate_one_folder(pred_folder:str,
                        only_L3:bool=False,
                        use_mp:bool=False,
                        invert:bool=False,
                        kmeans:bool=False,
                    ):
    pred_files = sorted([osp.join(pred_folder, file)
                         for file in os.listdir(pred_folder)
                         if file.endswith('.mha')])
    gt_files = [search_mha_file(GT_FOLDERS_PRIORITY_ORIGINAL_ENGINEERSORT, seriesUID, 'label') 
                for seriesUID in [
                    Path(file).stem for file in pred_files if file.endswith('.mha')
                ]]
    L3_slicess = find_L3_slices([Path(files).stem for files in pred_files])
    
    metric_list = []

    if use_mp:
        with mp.Pool(12) as p:
            results = []
            for gt_path, pred_path, L3_slices in zip(gt_files, pred_files, L3_slicess):
                result = p.apply_async(
                    calculate_one_pair,
                    args=(gt_path, pred_path, only_L3, invert, kmeans, L3_slices))
                results.append(result)

            for result in tqdm(results, 
                               desc='Evaulating',
                               total=len(gt_files),
                               dynamic_ncols=True,
                               leave=False):
                out = result.get()
                if out is not None:
                    metric_list.append(out)

    else:
        for gt_path, pred_path, L3_slices in tqdm(zip(gt_files, pred_files, L3_slicess),
                                       desc='Evaulating',
                                       total=len(gt_files),
                                       dynamic_ncols=True,
                                       leave=False):
            out = calculate_one_pair(gt_path, pred_path, only_L3, invert, kmeans, L3_slices)
            if out is not None:
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
    parser.add_argument('--kmeans', action='store_true', default=False)
    return parser.parse_args()




if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    args = parser_args()

    if args.kmeans is True:
        class_map = CLASS_MAP_AFTER_KMEANS
    else:
        class_map = CLASS_MAP
    
    # 执行评估
    # metric_list中元素的数量是样本数量
    # 每个元素是一个字典，包含四个指标的四个类的值和自身的SeriesUID
    metric_list = evaluate_one_folder(pred_folder=args.pred_root,
                                      only_L3=not args.whole_series,
                                      invert=args.invert,
                                      use_mp=args.mp,
                                      kmeans=args.kmeans)
    
    # 汇总整理
    result = pd.DataFrame(metric_list).dropna().drop_duplicates('seriesUID')
    # 将一个metric分解写入为Class-Wise-Column
    for metric_column in result.drop(columns='seriesUID').columns:
        colume_names = [f"{metric_column}_{class_map[i]}" for i in range(1, len(class_map))]
        result[colume_names] = pd.DataFrame(result[metric_column].tolist(), index=result.index)
        result = result.drop(columns=metric_column)
        # 计算这一metric的类平均值
        result[f'{metric_column}_Avg'] = result[colume_names].mean(axis=1)
    
    # 保存csv
    result.to_csv(os.path.join(args.pred_root, 'evaluation.csv'), index=False)
    print(result)
