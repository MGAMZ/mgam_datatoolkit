from ..base import mgam_Standard_3D_Mha, mgam_Standard_Precropped_Npz
from .meta import CLASS_INDEX_MAP



class FLARE_2023_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class FLARE_2023_Precrop_Npz(FLARE_2023_base, mgam_Standard_Precropped_Npz):
    pass


class FLARE_2023_Mha(FLARE_2023_base, mgam_Standard_3D_Mha):
    pass