import os
import pickle
import pdb

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from mmcv.transforms import BaseTransform
from mmengine.logging import print_log, MMLogger
from mmpretrain.datasets.base_dataset import BaseDataset
from mmpretrain.structures import DataSample

# 有两列是中文标注，映射到数字索引
TISSUE_CLASS_INDEX = {
    "default": 0,
    "乳头型": 1,
    "实性型": 2,
    "贴壁型": 3,
    "微乳头型": 4,
    "腺泡型": 5,
}
COMBINATION_LABELS = {
    (12,14): "标签12+14",
    (12,16): "标签12+16",
    (14,16): "标签14+16",
    (12,14,16): "标签12+14+16",
}
EXCLUDE_CLASSES = [
    "date_dx",
    "date_lfp",
    "date_relapse",
    "date_mt",
    "id",
    "蜡块号",
    "location", # AR样本中不存在这一标注
    "OS_mo",    # AR样本中不存在这一标注
    "death",    # AR样本中不存在这一标注
    *COMBINATION_LABELS.values()
]
# 定义标签序号和他们隶属的相关组别
LABEL_GROUP = {
    "Immunization": [1,2,3,4,5,6,7],        # 免疫相关标注
    "Histological": [8,],                   # 组织学形态
    "TumorNucleus": [9,10,11,12,13,14],     # 肿瘤细胞核
    "TumorStroma":  [15,16,],               # 肿瘤间质
    "GenePhenoTp":  [17,18,],               # 基因表型
}


class LabelCsvProvider:
    SPLIT_RATIO = [0.7, 0.3]
    
    def __init__(self, csv_path: str, split: str):
        self.csv_path = csv_path
        self.split = split
        self._load()
        self.group_index = self._parse_group_sample()
        self.label_names = [i for i in self.df.columns if "标签" in i]
        self.input_names = self.df.drop(columns=["实验室编号"] + self.label_names).columns.tolist()
        self._normalize()

    def _load(self):
        anno = pd.read_excel(self.csv_path, sheet_name="清洗")
        
        # 筛选AR的子标注集，因为我们的数据暂时只有AR的样本。
        anno = anno[anno['实验室编号'].str.contains('AR').fillna(False)]
        
        # 排除所有无关列
        anno = anno.drop(columns=EXCLUDE_CLASSES)
        
        # 按照split和ratio切分
        split_idx = int(len(anno) * self.SPLIT_RATIO[0])
        if self.split == "train":
            anno = anno.iloc[:split_idx]
        else:
            anno = anno.iloc[split_idx:]
        
        # 将中文标注映射到数字索引
        for ori, idx in TISSUE_CLASS_INDEX.items():
            anno = anno.replace(ori, idx)
        
        # 各行缺失值填充到各自的平均值，仅对input_names有效
        label_names = [i for i in anno.columns if "标签" in i]
        input_names = anno.drop(columns=["实验室编号"] + label_names).columns.tolist()
        for col in input_names:
            anno[col] = anno[col].fillna(anno[col].mean())
        
        self.df = anno

    def _normalize(self):
        # 对输入特征进行归一化
        scaler = MinMaxScaler()
        self.df[self.input_names] = scaler.fit_transform(self.df[self.input_names])

    def _parse_group_sample(self):
        # 对于不同的标签子集，找到它们可用的样本
        # 这是因为有些子集的样本特别对某些特征进行了标注
        # 而不属于这个子集的样本则大量缺失该通道的标注
        # 因此在后面建模的时候应该是需要单独处理的
        group_samples = {}
        for group_name, label_nums in LABEL_GROUP.items():
            label_cols = [f"标签{num}" for num in label_nums]
            valid_df = self.df.dropna(subset=label_cols)
            raw_ids = valid_df["实验室编号"].unique().tolist()
            
            # 存在一些属于同种标注的不同slide样本
            # 在label中，它们的实验室编号为这种形式：“AR017/AR087”
            processed_ids = []
            for lab_id in raw_ids:
                if isinstance(lab_id, str) and '/' in lab_id:
                    # 分割包含"/"的编号并添加到列表
                    split_ids = lab_id.split('/')
                    processed_ids.extend(split_ids)
                else:
                    processed_ids.append(lab_id)
            
            # 去重
            group_samples[group_name] = list(dict.fromkeys(processed_ids))
            
        return group_samples

    def get_sample_groups(self, lab_id: str) -> list:
        """
        根据实验室编号返回其所属的标签组别列表
        
        Args:
            lab_id (str): 实验室编号，如 'AR017'
            
        Returns:
            list: 该样本所属的标签组别列表
        """
        belonging_groups = []
        for group_name, samples in self.group_index.items():
            if lab_id in samples:
                belonging_groups.append(group_name)
        return belonging_groups
    
    def get_sample_labels(self, lab_id: str) -> dict|None:
        """
        获取指定样本在其所属组别中的所有标签值
        
        Args:
            lab_id (str): 实验室编号
            
        Returns:
            dict: 按组别组织的标签值字典
                {
                    "group_name": [label1, label2, ...],
                    ...
                }
        """
        belonging_groups = self.get_sample_groups(lab_id)
        if len(belonging_groups) == 0:
            return None
        result = {}
        
        # 找到对应lab_id的行
        sample_row = self.df[self.df['实验室编号'].str.contains(lab_id, na=False)].iloc[0]
        
        # 对每个组别获取其标签值
        for group in belonging_groups:
            label_nums = LABEL_GROUP[group]
            labels = []
            for num in label_nums:
                label_value = sample_row[f"标签{num}"]
                labels.append(label_value)
            result[group] = np.array(labels).astype(np.int64)
            
        return result
    
    def get_sample_feature(self, lab_id: str) -> np.ndarray|None:
        """
        获取指定样本的特征值
        
        Args:
            lab_id (str): 实验室编号
            
        Returns:
            dict: 该样本的特征值字典
        """
        belonging_groups = self.get_sample_groups(lab_id)
        if len(belonging_groups) == 0:
            return None
        sample_row = self.df[self.df['实验室编号'].str.contains(lab_id, na=False)].iloc[0]
        # 丢弃包含标签的列，剩下的都是输入列
        sample_row = sample_row.drop(labels=["实验室编号"] + self.label_names).to_numpy()
        return sample_row.astype(np.float32)


class CLAM_Feat(BaseDataset):
    SPLIT_RATIO = [0.7, 0.3, 0.0]
    
    def __init__(self, csv_path:str, split:str, debug, *args, **kwargs):
        self.label_backend = LabelCsvProvider(csv_path, split)
        self.split = split
        self.debug = debug
        self.data_root:str
        super(CLAM_Feat, self).__init__(ann_file="", *args, **kwargs)

    def _parse_sample_id(self, pt_file_path:str) -> str:
        assert pt_file_path.endswith(".pt")
        return os.path.basename(pt_file_path).split('-')[0]

    def load_data_list(self) -> list[dict]:
        pts = [os.path.join(self.data_root, i) 
               for i in os.listdir(self.data_root) 
               if i.endswith(".pt")]
        
        # OpenMM接口
        # 将注释表中的每一行转换为样本字典，包括CSV特征、.pt文件路径和标签信息
        samples = []
        for pt_path in pts:
            id = self._parse_sample_id(pt_path)
            labels = self.label_backend.get_sample_labels(id)
            feat_anno = self.label_backend.get_sample_feature(id)
            if labels is None or feat_anno is None:
                continue
            
            sample = {
                "pt_path": pt_path, # file path of Tensor [num_patches, channels]
                "feat_anno": feat_anno, # [num_feat, ]
                "gt_label": labels  # {group_name: [label1, label2, ...]}
            }
            samples.append(sample)

        if self.debug:
            samples = samples[:32]
        print_log(
            f"WSL Dataset ({self.split}) loaded {len(samples)} samples.",
            MMLogger.get_current_instance(),
        )
        return samples if not self.debug else samples[:10]


class Label_for_SVM(BaseTransform):
    def transform(self, results:dict):
        l = results["gt_score"]
        l[l <0.5] = -1
        l[l >=0.5] = 1
        results["gt_score"] = l
        return results


"""
sample = {
    "pt_path": pt_path, 
    "csv_feat": csv_feat, 
    "gt_label": labels  # {group_name: [label1, label2, ...]}
}
"""


class LoadPt(BaseTransform):
    def transform(self, results:dict):
        try:
            # [num_patches, channels]
            results["pt"] = torch.load(results["pt_path"], weights_only=False).float()
        except Exception as e:
            raise RuntimeError(f"Failed to load feature from {results['pt_path']}") from e
        
        return results


class FudanDoubleBlindSample(DataSample):
    def to(self, *args, **kwargs):
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
            elif isinstance(v, dict):
                new_data.set_field(
                    {kk: vv.to(*args, **kwargs) 
                     for kk, vv in v.items()},
                    k
                )
        return new_data


class Pack(BaseTransform):
    @staticmethod
    def to_tensor(array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array)
        elif isinstance(array, dict):
            return {k: Pack.to_tensor(v) for k, v in array.items()}
    
    def transform(self, results:dict):
        sample = FudanDoubleBlindSample()
        # {group_name: [label1, label2, ...]}
        sample.set_field(self.to_tensor(results['gt_label']), "gt_label")
        sample.set_field(self.to_tensor(results['feat_anno']), "feat_anno")
        
        """
        inputs: torch.Tensor [num_patches, channels]
        data_sample: DataSample
            - gt_label: dict {group_name: [label1, label2, ...]}
            - feat_anno: np.ndarray [num_feat, ]
        """
        return {
            "inputs": results["pt"],
            "data_samples": sample
        }