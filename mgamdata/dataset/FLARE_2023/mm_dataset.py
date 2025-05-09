from ..base import mgam_SemiSup_3D_Mha, mgam_SemiSup_Precropped_Npz, mgam_SemiSup_3D_Mha, mgam_Patched_Mha
from .meta import CLASS_INDEX_MAP



class FLARE_2023_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class FLARE_2023_Precrop_Npz(FLARE_2023_base, mgam_SemiSup_Precropped_Npz):
    pass


class FLARE_2023_Patched_Mha(FLARE_2023_base, mgam_Patched_Mha):
    pass


class FLARE_2023_Mha(FLARE_2023_base, mgam_SemiSup_3D_Mha):
    pass


class FLARE_2023_Semi_Mha(FLARE_2023_base, mgam_SemiSup_3D_Mha):
    pass