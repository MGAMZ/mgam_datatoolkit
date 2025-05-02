from ..base import mgam_Standard_3D_Mha, mgam_SemiSup_Precropped_Npz, mgam_SemiSup_3D_Mha
from .meta import CLASS_INDEX_MAP



class FLARE_2022_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class FLARE_2022_Precrop_Npz(FLARE_2022_base, mgam_SemiSup_Precropped_Npz):
    pass


class FLARE_2022_Mha(FLARE_2022_base, mgam_Standard_3D_Mha):
    pass


class FLARE_2022_Semi_Mha(FLARE_2022_base, mgam_SemiSup_3D_Mha):
    pass