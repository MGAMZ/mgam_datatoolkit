import os
import pdb
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk


def find_L3_slices(seriesUIDs: list[str]|str,
                   L3_df_path:str|pd.DataFrame,
                   attrs:list[str]=['L3节段起始层数','L3节段终止层数']):
    if isinstance(seriesUIDs, str):
        seriesUIDs = [seriesUIDs]
    if isinstance(L3_df_path, str):
        L3_df = pd.read_excel(L3_df_path)
    elif isinstance(L3_df_path, pd.DataFrame):
        L3_df = L3_df_path
    else:
        raise RuntimeError(f"received invalid L3_df:{L3_df_path}, type {type(L3_df_path)}")
    
    L3_slicess = []
    
    for seriesUID in seriesUIDs:
        found_items = L3_df['序列编号'] == seriesUID
        series_anno = L3_df.loc[found_items]
        if found_items.sum() > 1:
            series_anno = series_anno.iloc[0]
            warnings.warn(f"More than one annotation for {seriesUID}, using the first one.")
        series_L3 = series_anno[attrs].values.flatten().astype(np.float32)
        
        if np.isnan(series_L3).any() or len(series_L3) == 0:
            L3_slicess.append(None)
        else:
            L3_slicess.append(series_L3.astype(np.uint32))
    
    return L3_slicess


def resample_L3_anno(L3_xlsx_path:str, 
                     ori_itk_folder:str, 
                     target_itk_folder:str, 
                     anno_save_file_path:str):
    """
    根据原始和目标ITK文件的spacing比例，重采样L3标注信息。
    
    参数:
    L3_xlsx_path: L3标注Excel文件路径
    ori_itk_folder: 原始ITK文件夹路径
    target_itk_folder: 目标ITK文件夹路径
    anno_save_file_path: 重采样后标注保存的文件路径
    """
    anno = pd.read_excel(L3_xlsx_path)
    target_anno = pd.DataFrame(columns=anno.columns)
    
    # 遍历原始ITK文件夹中的所有mha文件
    for ori_file in os.listdir(ori_itk_folder):
        if ori_file.endswith('.mha'):
            # 获取SeriesUID（文件名去除后缀）
            series_uid = os.path.splitext(ori_file)[0]
            
            # 构建原始和目标ITK文件路径
            ori_itk_path = os.path.join(ori_itk_folder, ori_file)
            target_itk_path = os.path.join(target_itk_folder, ori_file)
            
            # 检查目标文件是否存在
            if not os.path.exists(target_itk_path):
                print(f"目标文件不存在：{target_itk_path}")
                continue
            
            # 使用find_L3_slices函数查找L3标注
            L3_slices = find_L3_slices(series_uid, anno, ['L3节段起始层数', 'L3节段椎弓根层面层数', 'L3节段终止层数'])
            
            if L3_slices[0] is None:
                print(f"SeriesUID {series_uid} 没有有效的L3标注, 输出文件中排除该文件")
                continue
            
            # 获取原始标注
            ori_start, ori_mid, ori_end = L3_slices[0]
            
            # 读取原始和目标ITK文件
            try:
                ori_itk = sitk.ReadImage(ori_itk_path)
                target_itk = sitk.ReadImage(target_itk_path)
            except Exception as e:
                print(f"读取文件出错：{e}")
                continue
            
            # 获取原始和目标文件的spacing
            ori_spacing = np.array(ori_itk.GetSpacing())
            target_spacing = np.array(target_itk.GetSpacing())
            
            # Z轴方向的spacing比例
            z_spacing_ratio = ori_spacing[2] / target_spacing[2]
            
            # 缩放标注层数
            resampled_start = np.round(ori_start * z_spacing_ratio).astype(np.uint16)
            resampled_mid = np.round(ori_mid * z_spacing_ratio).astype(np.uint16)
            resampled_end = np.round(ori_end * z_spacing_ratio).astype(np.uint16)
            
            # 复制条目
            mask = anno['序列编号'] == series_uid
            if mask.any():
                # 复制该行数据
                row_to_copy = anno.loc[mask].copy()
                # 更新重采样后的值
                row_to_copy['L3节段起始层数'] = int(resampled_start)
                row_to_copy['L3节段终止层数'] = int(resampled_end)
                row_to_copy['L3节段椎弓根层面层数'] = int(resampled_mid)
                # 将行添加到目标DataFrame
                target_anno = pd.concat([target_anno, row_to_copy], ignore_index=True)
    
    target_anno.to_excel(anno_save_file_path, index=False)
    print(f"重采样标注已保存到 {anno_save_file_path}")
    return target_anno
