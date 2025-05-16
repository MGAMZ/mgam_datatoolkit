import os
import pdb
from tqdm import tqdm
from colorama import Fore, Style

import numpy as np
import pandas as pd


# 读取Slide-Patch映射
class FileID_Map:
    def __init__(self, file_id_map_paths: list[str]):
        self.ann = []
        # 读取并合并所有映射文件
        for path in tqdm(file_id_map_paths, desc="加载映射文件", dynamic_ncols=True, leave=False):
            try:
                df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
                if "originPath" not in df.columns or "seriesinstanceUID" not in df.columns:
                    raise ValueError(f"CSV file {path} missing required columns")
                self.ann.append(df)
            except Exception as e:
                print(Fore.YELLOW, f"Error loading {path}: {str(e)}", Style.RESET_ALL)

        if not self.ann:
             print(Fore.YELLOW, "Warning: No valid data loaded from any mapping files. Initializing empty map.", Style.RESET_ALL)
             # 创建一个空的DataFrame，包含必要的列，以避免后续操作出错
             self.ann = pd.DataFrame(columns=["originPath", "seriesinstanceUID"])
        else:
            self.ann = pd.concat(self.ann, ignore_index=True)
            # 删除重复项
            self.ann = self.ann.drop_duplicates(subset=["originPath"], keep="last")

        print("[FileID_Map] HINT: Replacing slashes `/` in originPath with underscores '_' to align with the newest file name format.")
        # 确保在应用replace之前，列是字符串类型
        self.ann['originPath'] = self.ann['originPath'].astype(str).str.replace('/', '_')

    def search_from_file_path(self, file_path: str):
        # 确保比较时类型一致
        found_patch = self.ann[self.ann["originPath"] == str(file_path)]
        if found_patch.empty:
            return None
        else:
            # 确保返回的是字符串
            return str(found_patch["seriesinstanceUID"].values[0])

    def search_from_seriesUID(self, seriesUID:str):
         # 确保比较时类型一致
        found_patch = self.ann[self.ann["seriesinstanceUID"] == str(seriesUID)]
        if found_patch.empty:
            return None
        else:
             # 确保返回的是字符串
            return str(found_patch["originPath"].values[0])

# 读取细胞位置的标注
class PointReader:
    def __init__(self, anno_files: list[str]):
        self.ann = []
        # 读取并合并所有位置标注文件
        # 使用tqdm包装，提供加载进度
        for anno_file in tqdm(anno_files, desc="加载位置标注文件", dynamic_ncols=True, leave=False):
            try:
                df = pd.read_excel(anno_file) if anno_file.endswith('.xlsx') else pd.read_csv(anno_file)
                # 检查必要列是否存在
                if "序列编号" not in df.columns or "影像结果" not in df.columns:
                    raise ValueError(f"位置标注文件 {anno_file} 缺少必要的列: '序列编号' 或 '影像结果'")
                # 确保序列编号是字符串类型，以进行一致的比较
                df["序列编号"] = df["序列编号"].astype(str)
                # 合并DataFrame
                self.ann.append(df)
            except Exception as e:
                print(Fore.YELLOW, f"读取 {anno_file} 时出错: {str(e)}", Style.RESET_ALL)

        if not self.ann:
            print(Fore.YELLOW, "警告: 未能从任何位置标注文件中加载有效数据。初始化空标注。", Style.RESET_ALL)
            self.ann = pd.DataFrame(columns=["序列编号", "影像结果"])
        else:
            self.ann = pd.concat(self.ann, ignore_index=True)
            # 如果有重复的序列编号，保留所有标签
            # NOTE 这一步会aggregate影像结果为列表
            # 在groupby之前处理NaN值，避免它们被错误地聚合
            self.ann.dropna(subset=["序列编号", "影像结果"], inplace=True)
            self.ann = self.ann.groupby("序列编号")["影像结果"].agg(list).reset_index()

    def search_from_SeriesUID(self, SeriesUID: str):
        # 确保比较时类型一致
        found_labels = self.ann[self.ann["序列编号"] == str(SeriesUID)]["影像结果"]
        if found_labels.empty:
            return None
        else:
            # 返回标签列表
            return found_labels.values[0]

# 读取slide是否成团的标注
class ClusterReader:
    def __init__(self, anno_files: list[str]):
        self.ann = []
        for anno_file in anno_files:
            try:
                df = pd.read_excel(anno_file) if anno_file.endswith('.xlsx') else pd.read_csv(anno_file)
                # 验证必要列
                if "序列编号" not in df.columns or "是否成团" not in df.columns:
                    raise ValueError(f"分类标注文件 {anno_file} 缺少必要的列: '序列编号' 或 '是否成团'")
                # 确保序列编号是字符串类型
                df["序列编号"] = df["序列编号"].astype(str)
                # 合并数据
                self.ann.append(df)
            except Exception as e:
                print(Fore.YELLOW, f"读取 {anno_file} 时出错: {str(e)}", Style.RESET_ALL)

        if not self.ann:
             print(Fore.YELLOW, "警告: 未能从任何分类标注文件中加载有效数据。初始化空标注。", Style.RESET_ALL)
             # 创建一个空的DataFrame，包含必要的列
             self.ann = pd.DataFrame(columns=["序列编号", "是否成团"])
        else:
            self.ann = pd.concat(self.ann, ignore_index=True)
            # 处理重复记录，保留最新记录
            self.ann = self.ann.drop_duplicates(subset=["序列编号"], keep="last")

    def is_clustered(self, SeriesUID: str) -> bool | None:
        """
        查询指定序列是否成团
        Args:
            SeriesID: 序列编号
        Returns:
            bool: 成团状态
            None: 未找到记录
        """
         # 确保比较时类型一致
        found = self.ann[self.ann["序列编号"] == str(SeriesUID)]["是否成团"]
        if found.empty:
            return None
        # 处理可能的非布尔值或NaN
        val = found.values[0]
        if pd.isna(val):
            return None
        # 尝试将常见表示（如1/0, '是'/'否', 'True'/'False'）转换为布尔值
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, float, np.number)):
            return bool(val)
        if isinstance(val, str):
            val_lower = val.lower()
            if val_lower in ['true', '是', '1', 'yes']:
                return True
            if val_lower in ['false', '否', '0', 'no']:
                return False
        # 如果无法明确转换，返回None或打印警告
        print(Fore.YELLOW + f"警告: 无法将分类标注值 '{val}' 转换为布尔值 (SeriesUID: {SeriesUID})" + Style.RESET_ALL)
        return None

    def check_has_anno(self, SeriesUID:str):
        return not self.ann[self.ann["序列编号"] == str(SeriesUID)].empty
