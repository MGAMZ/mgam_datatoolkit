import os
import re

import numpy as np

from ..base import mgam_Standard_Patched_Npz
from .meta import CLASS_INDEX_MAP



class RoseThyroidCount_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys())[1:])


class RoseThyroidCount_Precrop_Npz(RoseThyroidCount_base, mgam_Standard_Patched_Npz):
    SPLIT_RATIO = [0.8, 0.2, 0]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
