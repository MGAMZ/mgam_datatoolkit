import os

from ..base import mgam_Standard_Patched_Npz
from .meta import CLASS_INDEX_MAP



class RoseThyroidCount_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class RoseThyroidCount_Precrop_Npz(RoseThyroidCount_base, mgam_Standard_Patched_Npz):
    SPLIT_RATIO = [0.8, 0.2]
    
    def __init__(self, *args, **kwargs):
        if kwargs['split'] == 'test':
            kwargs['split'] = 'val'
        super().__init__(*args, **kwargs)
