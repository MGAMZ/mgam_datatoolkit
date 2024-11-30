from ..base import mgam_Standard_3D_Mha, mgam_Standard_Precropped_Npz
from .meta import CLASS_INDEX_MAP



class AbdomenCT_1K_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class AbdomenCT_1K_Precrop_Npz(AbdomenCT_1K_base, mgam_Standard_Precropped_Npz):
    pass


class AbdomenCT_1K_Mha(AbdomenCT_1K_base, mgam_Standard_3D_Mha):
    pass