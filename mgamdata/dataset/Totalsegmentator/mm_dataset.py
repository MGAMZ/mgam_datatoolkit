import os
from os.path import join

from typing import List

from mmengine.logging import print_log, MMLogger
from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

from . import CLASS_INDEX_MAP



@DATASETS.register_module()
class TotalsegmentatorSegDataset(BaseDataset):
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

    def __init__(self, split:str, data_root:str, **kwargs) -> None:
        super().__init__(data_root=data_root, **kwargs)
        self.split = split
        self.data_root:str
    
    
    def fetch_samples(self):
        splited_image_folder = join(self.data_root, 'img_dir', self.split)
        splited_anno_folder = join(self.data_root, 'ann_dir', self.split)
        available_series = os.listdir(splited_image_folder)
        
        for series in available_series:
            series_folder = join(splited_image_folder, series)
            image_paths = sorted(os.listdir(series_folder))
            
            for image_path in image_paths:
                image_path = join(series_folder, image_path)
                anno_path = image_path.replace(splited_image_folder, splited_anno_folder)
                yield (image_path, anno_path)
    
    
    def load_data_list(self) -> List[dict]:
        """
        分割数据集的样本字典需要以下Key:
        - img_path: str, 图像路径
        - seg_map_path: str, 分割标签路径
        - label_map: str, 分割标签的类别映射，默认为空。它是矫正映射，如果map没有问题，则不需要矫正。
        - reduce_zero_label: bool, 是否将分割标签中的0类别映射到-1(255), 默认为False
        - seg_fields: list, 分割标签的字段名, 默认为空列表
        """
        data_list = []
        for image_path, anno_path in self.fetch_samples():
            data_list.append(dict(
                img_path=image_path,
                seg_map_path=anno_path,
                label_map=None,
                reduce_zero_label=False,
                seg_fields=[]
            ))
        print_log(f"Totalsegmentator dataset {self.split} split loaded {len(data_list)} samples.",
                  MMLogger.get_current_instance())
        return sorted(data_list, key=lambda x: x['img_path'])
