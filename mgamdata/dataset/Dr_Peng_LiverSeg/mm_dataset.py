from ..base import mgam_SemiSup_3D_Mha, mgam_SemiSup_Precropped_Npz
from .meta import CLASS_INDEX_MAP



class LiverSeg_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class LiverSeg_Precrop_Npz(LiverSeg_base, mgam_SemiSup_Precropped_Npz):
    pass


class LiverSeg_Semi_Mha(LiverSeg_base, mgam_SemiSup_3D_Mha):
    pass
