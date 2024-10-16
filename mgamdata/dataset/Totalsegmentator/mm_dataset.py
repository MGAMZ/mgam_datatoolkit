from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS

from . import CLASS_INDEX_MAP



@DATASETS.register_module()
class TotalsegmentatorDataset(BaseSegDataset):
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

    def __init__(self,
                 img_suffix='.tiff',
                 seg_map_suffix='.tiff',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
