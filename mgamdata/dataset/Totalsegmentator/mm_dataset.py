import os
from os.path import join
from tqdm import tqdm

import orjson
import pandas as pd

from mmcv.transforms import BaseTransform
from mmengine.logging import print_log, MMLogger
from mmseg.datasets.basesegdataset import BaseSegDataset
from tqdm import tqdm

from .meta import (
    CLASS_INDEX_MAP, DATA_ROOT_SLICE2D_TIFF, 
    get_subset_and_rectify_map, DATA_ROOT_3D_MHA, META_CSV_PATH)



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



class TotalsegmentatorSegDataset(BaseSegDataset):
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

    def __init__(self,
                 split:str,
                 subset:str|None=None,
                 debug:bool=False,
                 **kwargs) -> None:
        self.split = split
        self.data_root = DATA_ROOT_SLICE2D_TIFF
        self.indexer = TotalsegmentatorIndexer(self.data_root)
        self.debug = debug
        
        if subset is not None:
            new_classes = list(get_subset_and_rectify_map(subset)[0].keys())
        else:
            new_classes = self.METAINFO['classes']
        
        super().__init__(data_root=self.data_root, 
                         metainfo={'classes':new_classes}, 
                         **kwargs)


    def _update_palette(self) -> list[list[int]]:
        '''确保background为RGB全零'''
        new_palette = super()._update_palette()
        return [[0,0,0]] + new_palette[1:]


    def load_data_list(self):
        """
        Sample Required Keys in mmseg:
        
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
        sorted_samples = sorted(data_list, key=lambda x: x['img_path'])
        
        if self.debug:
            return sorted_samples[:16]
        else:
            return sorted_samples



class TotalsegmentatorSeg3DDataset(BaseSegDataset):
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

    def __init__(self,
                 split:str,
                 subset:str|None=None,
                 debug:bool=False,
                 **kwargs) -> None:
        self.split = split
        self.debug = debug
        self.meta_table = pd.read_csv(META_CSV_PATH)
        
        if subset is not None:
            new_classes = list(get_subset_and_rectify_map(subset)[0].keys())
        else:
            new_classes = self.METAINFO['classes']
        
        super().__init__(metainfo={'classes':new_classes}, **kwargs)


    def _update_palette(self) -> list[list[int]]:
        '''确保background为RGB全零'''
        new_palette = super()._update_palette()
        return [[0,0,0]] + new_palette[1:]


    def iter_series(self):
        activate_series = self.meta_table[self.meta_table['split'] == self.split]
        activate_series_id = activate_series['image_id'].tolist()
        self.data_root: str
        for series in activate_series_id:
            yield (os.path.join(self.data_root, series, 'ct.mha'),
                   os.path.join(self.data_root, series, 'segmentations.mha'))


    def load_data_list(self):
        """
        Sample Required Keys in mmseg:
        
        - img_path: str, 图像路径
        - seg_map_path: str, 分割标签路径
        - label_map: str, 分割标签的类别映射，默认为空。它是矫正映射，如果map没有问题，则不需要矫正。
        - reduce_zero_label: bool, 是否将分割标签中的0类别映射到-1(255), 默认为False
        - seg_fields: list, 分割标签的字段名, 默认为空列表
        """
        data_list = []
        for image_path, anno_path in self.iter_series():
            data_list.append(dict(
                img_path=image_path,
                seg_map_path=anno_path,
                label_map=None,
                reduce_zero_label=False,
                seg_fields=[]
            ))
        
        print_log(f"Totalsegmentator dataset {self.split} split loaded {len(data_list)} samples.",
                  MMLogger.get_current_instance())
        sorted_samples = sorted(data_list, key=lambda x: x['img_path'])
        
        if self.debug:
            return sorted_samples[:16]
        else:
            return sorted_samples



class ParseID(BaseTransform):
    def transform(self, results):
        results['series_id'] = os.path.basename(
            os.path.dirname(results['img_path'])
        )
        return results


