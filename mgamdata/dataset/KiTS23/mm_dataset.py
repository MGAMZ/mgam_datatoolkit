from ..base import mgam_SemiSup_Precropped_Npz, mgam_SemiSup_3D_Mha
from .meta import CLASS_INDEX_MAP



class KiTS23_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class KiTS23_Precrop_Npz(KiTS23_base, mgam_SemiSup_Precropped_Npz):
    pass


class KiTS23_Mha(KiTS23_base, mgam_SemiSup_3D_Mha):
    pass