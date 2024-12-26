import os
import re

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
        all_series = sorted(
            os.listdir(self.data_root),
            key=lambda x: abs(int(re.search(r"\d+", x).group())),
        )
        assert self.TEST_SLIDE_UID in all_series, f"Missing Test Slide {self.TEST_SLIDE_UID}."
        all_series.remove(self.TEST_SLIDE_UID)

        if self.split == "test":
            return [self.TEST_SLIDE_UID, ]
        elif self.split == "val" or self.split == "train":
            return all_series
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")
