import os
import re
import pdb
import numpy as np

from mmcv.transforms import BaseTransform
from ..base import mgam_Standard_Patched_Npz
from .meta import CLASS_INDEX_MAP


class RoseThyroidCount_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class RoseThyroidCount_Precrop_Npz(RoseThyroidCount_base, mgam_Standard_Patched_Npz):
    TEST_SLIDE_UID = "5cc71dcf6292dedec40940f26f4c5cdfdc39c4be"

    def __init__(self, *args, **kwargs):
        if kwargs["split"] == "test":
            kwargs["split"] = "val"
        super().__init__(*args, **kwargs)

    def _split(self):
        all_series = [i 
                      for i in os.listdir(self.data_root) 
                      if os.path.isdir(os.path.join(self.data_root, i))]
        assert self.TEST_SLIDE_UID in all_series, f"Missing Test Slide {self.TEST_SLIDE_UID}."
        all_series.remove(self.TEST_SLIDE_UID)

        if self.split == "test":
            return [self.TEST_SLIDE_UID, ]
        elif self.split == "val" or self.split == "train":
            return all_series
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")



class Normalizer_cell2(BaseTransform):
    # RGB order
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def transform(self, results:dict):
        results['img'] = (results['img']/255 - self.mean) / self.std
        return results


class BGR2RGB(BaseTransform):
    def transform(self, results:dict):
        results['img'] = results['img'][..., ::-1]
        return results
