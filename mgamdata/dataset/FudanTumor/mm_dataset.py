import os
import pickle
import pdb

import torch
import numpy as np
import pandas as pd

from mmpretrain.datasets.base_dataset import BaseDataset


TISSUE_CLASS_INDEX = {
    "default": 0, 
    "乳头型": 1, 
    "实性型": 2, 
    "贴壁型": 3, 
    "微乳头型": 4, 
    "腺泡型": 5,
}


class WSL_Dataset(BaseDataset):
    ID_COLUMN = 'slide_id'
    SPLIT_RATIO = [0.7, 0.1, 0.2]
    
    def __init__(self, data_root:str, anno:str, split:str, debug:bool, *args, **kwargs):
        self.pt_sample_list = os.listdir(data_root)
        self.anno = pd.read_excel(anno)
        self.split = split
        self.debug = debug
        self._split()
        self.anno.fillna(0, inplace=True)
        super(WSL_Dataset, self).__init__(ann_file='', data_root=data_root, *args, **kwargs)
    
    def _split(self):
        train_size = int(len(self.anno)*self.SPLIT_RATIO[0])
        val_size = int(len(self.anno)*self.SPLIT_RATIO[1])
        
        if self.split == 'train':
            self.anno = self.anno.iloc[:train_size]
        elif self.split == 'val':
            self.anno = self.anno.iloc[train_size:train_size+val_size]
        elif self.split == 'test':
            self.anno = self.anno.iloc[train_size+val_size:]
        else:
            raise ValueError(f"Invalid split mode: {self.split}")
        
    def _parse_sample(self, sample:pd.Series) -> tuple:
        # 寻找列包含“标签”的列，并取出该部分数据，组成np.ndarray
        labels = sample.filter(like='标签').values
        # 剩下的列全部都是输入通道
        label_keys = list(sample.filter(like='标签').keys())
        inputs = sample.drop(labels=label_keys + ['slide_id']).values
        
        # 中文标注映射到唯一index
        for key, value in TISSUE_CLASS_INDEX.items():
            inputs[inputs == key] = value
            labels[labels == key] = value
        
        return inputs.astype(np.float32), labels.astype(np.float32)
    
    def _load_features(self, slide_id:str):
        for pt_sample in self.pt_sample_list:
            if slide_id in pt_sample:
                return os.path.join(self.data_root, pt_sample)
        else:
            return None
    
    def load_data_list(self) -> list[dict]:
        samples = []
        for _, csv_sample in self.anno.iterrows():
            slide_id = csv_sample[self.ID_COLUMN]
            inputs, labels = self._parse_sample(csv_sample.drop(columns=[self.ID_COLUMN]))
            features = self._load_features(slide_id)
            if features is None:
                continue
            samples.append({
                "feat_csv": inputs, 
                "feat_pt_path": features, 
                "gt_score": labels,
            })
        
        return samples if not self.debug else samples[:10]
