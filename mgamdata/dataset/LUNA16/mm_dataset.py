from ..base import mgam_Standard_3D_Mha, mgam_Standard_Precropped_Npz
from .meta import CLASS_INDEX_MAP



class LUNA16_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class LUNA16_Precrop_Npz(LUNA16_base, mgam_Standard_Precropped_Npz):
    pass


class LUNA16_Mha(LUNA16_base, mgam_Standard_3D_Mha):
    pass