import os
import re
import pdb
from abc import abstractmethod
from collections.abc import Generator, Iterable

import numpy as np

from mmcv.transforms import BaseTransform
from mmengine.logging import print_log, MMLogger
from mmseg.datasets.basesegdataset import BaseSegDataset



class ParseID(BaseTransform):
    def transform(self, results):
        results['series_id'] = os.path.basename(
            os.path.dirname(results['img_path'])
        )
        return results


class mgam_BaseSegDataset(BaseSegDataset):
    SPLIT_RATIO = [0.7, 0.15, 0.15]
    
    def __init__(self,
                 split:str,
                 debug:bool=False,
                 **kwargs) -> None:
        self.split = split
        self.debug = debug
        super().__init__(**kwargs)
        self.data_root: str

    def _update_palette(self) -> list[list[int]]:
        '''确保background为RGB全零'''
        new_palette = super()._update_palette()
        
        if len(self.METAINFO) > 1:
            return [[0,0,0]] + new_palette[1:]
        else:
            return new_palette

    @abstractmethod
    def sample_iterator(self
            ) -> Generator[tuple[str, str], None, None] | Iterable[tuple[str, str]]:
        ...

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
        for image_path, anno_path in self.sample_iterator():
            data_list.append(dict(
                img_path=image_path,
                seg_map_path=anno_path,
                label_map=self.label_map,
                reduce_zero_label=False,
                seg_fields=[],
            ))
        
        print_log(f"{self.__class__.__name__} dataset {self.split} split loaded {len(data_list)} samples.",
                  MMLogger.get_current_instance())
        
        if self.debug:
            return data_list[:16]
        else:
            return data_list


class mgam_Standard_3D_Mha(mgam_BaseSegDataset):
    def __init__(self,
                 data_root_mha:str,
                 **kwargs) -> None:
        self.data_root_mha = data_root_mha
        super().__init__(**kwargs)
        self.data_root: str

    def _split(self):
        all_series = [file.replace('.mha', '') 
                      for file in os.listdir(os.path.join(self.data_root_mha, 'label'))
                      if file.endswith('mha')]
        all_series = sorted(all_series, key=lambda x: abs(int(re.search(r'\d+', x).group())))
        np.random.shuffle(all_series)
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

    def sample_iterator(self) -> Generator[tuple[str, str], None, None]:
        for series in self._split():
            yield (os.path.join(self.data_root, 'image', series+'.mha'),
                   os.path.join(self.data_root, 'label', series+'.mha'))


class mgam_Standard_Npz_Structure:
    def sample_iterator(self) -> Generator[tuple[str, str], None, None]:
        for series in self._split():
            series_folder:str = os.path.join(self.data_root, series)
            for sample in os.listdir(series_folder):
                if sample.endswith('.npz'):
                    yield(os.path.join(series_folder, sample),
                          os.path.join(series_folder, sample))


class mgam_Standard_Precropped_Npz(mgam_Standard_Npz_Structure, mgam_Standard_3D_Mha):
    ...


class mgam_Standard_Patched_Npz(mgam_Standard_Npz_Structure, mgam_BaseSegDataset):
    def _split(self):
        all_series = [file.replace('.mha', '') 
                      for file in os.listdir(self.data_root)]
        all_series = sorted(all_series, key=lambda x: abs(int(re.search(r'\d+', x).group())))
        np.random.shuffle(all_series)
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