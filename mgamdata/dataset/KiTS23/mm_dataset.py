from ..base import mgam_Standard_3D_Mha, mgam_Standard_Precropped_Npz
from .meta import CLASS_INDEX_MAP



class KiTS23_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class KiTS23_Precrop_Npz(KiTS23_base, mgam_Standard_Precropped_Npz):
    pass


class KiTS23_Mha(KiTS23_base, mgam_Standard_3D_Mha):
    pass