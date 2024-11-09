import os
from abc import abstractmethod
from collections.abc import Generator, Iterable

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
        
        print_log(f"{self.__qualname__} dataset {self.split} split loaded {len(data_list)} samples.",
                  MMLogger.get_current_instance())
        
        if self.debug:
            return data_list[:16]
        else:
            return data_list
