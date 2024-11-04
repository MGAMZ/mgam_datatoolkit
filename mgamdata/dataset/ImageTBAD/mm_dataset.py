import os
import random
import re
from torch import Value
from tqdm import tqdm

import pandas as pd

from mmcv.transforms import BaseTransform
from mmengine.logging import print_log, MMLogger
from mmseg.datasets.basesegdataset import BaseSegDataset

from .meta import CLASS_INDEX_MAP, DATA_ROOT_3D_MHA


class ImageTBAD_Seg3DDataset(BaseSegDataset):
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))
    SPLIT_RATIO = [0.7, 0.15, 0.15]

    def __init__(self,
                 split:str,
                 debug:bool=False,
                 **kwargs) -> None:
        self.split = split
        self.debug = debug
        super().__init__(**kwargs)

    def _update_palette(self) -> list[list[int]]:
        '''确保background为RGB全零'''
        new_palette = super()._update_palette()
        return [[0,0,0]] + new_palette[1:]

    def _split(self):
        all_series = [file.replace('.mha', '') 
                      for file in os.listdir(os.path.join(DATA_ROOT_3D_MHA, 'label'))
                      if file.endswith('.mha')]
        all_series = sorted(all_series, key=lambda x: abs(int(re.search(r'\d+', x).group())))
        random.shuffle(all_series)
        total = len(all_series)
        train_end = int(total * self.SPLIT_RATIO[0])
        val_end = train_end + int(total * self.SPLIT_RATIO[1])
        
        if self.split == 'train':
            return all_series[:train_end]
        elif self.split == 'val':
            return all_series[train_end:val_end]
        elif self.split == 'test':
            return all_series[val_end:]
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")

    def iter_series(self):
        for series in self._split():
            yield (os.path.join(self.data_root, 'image', series),
                   os.path.join(self.data_root, 'label', series),)

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
        
        print_log(f"ImageTBAD dataset {self.split} split loaded {len(data_list)} samples.",
                  MMLogger.get_current_instance())
        
        if self.debug:
            return data_list[:16]
        else:
            return data_list


class TBAD_3Dnpz(ImageTBAD_Seg3DDataset):
    def iter_series(self):
        for series in self._split():
            series_folder = os.path.join(self.data_root, series)
            for sample in os.listdir(series_folder):
                yield(os.path.join(series_folder, sample),
                      os.path.join(series_folder, sample))


class ParseID(BaseTransform):
    def transform(self, results):
        results['series_id'] = os.path.basename(
            os.path.dirname(results['img_path'])
        )
        return results