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

    def _split(self):
        all_series = [i 
                      for i in os.listdir(self.data_root) 
                      if os.path.isdir(os.path.join(self.data_root, i))]
        assert self.TEST_SLIDE_UID in all_series, f"Missing Test Slide {self.TEST_SLIDE_UID}."
        all_series.remove(self.TEST_SLIDE_UID)

        if self.split == "test":
            return [self.TEST_SLIDE_UID, ]
        elif self.split == "train":
            return all_series[:-2]
        elif self.split == "val":
            return all_series[-2:]
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")
