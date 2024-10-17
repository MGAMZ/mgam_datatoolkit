import os
from os.path import join
from typing import List

import orjson

from mmengine.logging import print_log, MMLogger
from mmengine.dataset import BaseDataset
from tqdm import tqdm

try:
    from . import CLASS_INDEX_MAP
except:
    from mgamdata.dataset.Totalsegmentator.meta import CLASS_INDEX_MAP



class TotalsegmentatorIndexer:
    def __init__(self, data_root:str):
        self.data_root = data_root
        self.index_file = join(self.data_root, f'index.json')
        
        if not os.path.exists(self.index_file):
            self.generate_index_json_file()
        with open(self.index_file, 'rb') as f:
            self.img_index = orjson.loads(f.read())
    
    
    def generate_index_json_file(self):
        index = {
            split: list(self._index(join(self.data_root, 'img_dir'), split))
            for split in ['train', 'val', 'test']
        }
        with open(self.index_file, 'wb') as f:
            f.write(orjson.dumps(index, option=orjson.OPT_INDENT_2))
    
    
    def _index(self, image_root:str, split:str):
        split_folder = join(image_root, split)
        for series in tqdm(iterable = os.listdir(split_folder),
                           desc=f"Indexing {split} split",
                           dynamic_ncols=True,
                           leave=False):
            series_folder = join(split_folder, series)
            image_paths = sorted(os.listdir(series_folder))
            for image_path in image_paths:
                image_path = join(series_folder, image_path)
                yield os.path.relpath(image_path, self.data_root)


    def fetcher(self, split:str):
        selected_split_image_paths:list = self.img_index[split]
        return [
            (
                os.path.join(self.data_root, image_path),
                os.path.join(self.data_root, image_path.replace('img_dir', 'ann_dir'))
            )
            for image_path in selected_split_image_paths
        ]



class TotalsegmentatorSegDataset(BaseDataset):
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

    def __init__(self, split:str, data_root:str, path_index:str|None=None, **kwargs) -> None:
        self.path_index = path_index
        self.split = split
        self.data_root:str
        self.indexer = TotalsegmentatorIndexer(data_root)
        
        super().__init__(data_root=data_root, **kwargs)
    

    def load_data_list(self):
        """
        分割数据集的样本字典需要以下Key:
        - img_path: str, 图像路径
        - seg_map_path: str, 分割标签路径
        - label_map: str, 分割标签的类别映射，默认为空。它是矫正映射，如果map没有问题，则不需要矫正。
        - reduce_zero_label: bool, 是否将分割标签中的0类别映射到-1(255), 默认为False
        - seg_fields: list, 分割标签的字段名, 默认为空列表
        """
        data_list = []
        for image_path, anno_path in self.indexer.fetcher(self.split):
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



if __name__ == '__main__':
    from mgamdata.dataset.Totalsegmentator.meta import DATA_ROOT
    dataset = TotalsegmentatorSegDataset(
        'train', 
        join(DATA_ROOT, 'Totalsegmentator_dataset_v201_OpenmmTIFF')
    )
    for sample in dataset.load_data_list():
        print(sample)
