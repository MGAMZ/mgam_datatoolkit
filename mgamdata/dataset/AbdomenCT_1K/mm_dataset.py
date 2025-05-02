from ..base import mgam_SemiSup_3D_Mha, mgam_SemiSup_Precropped_Npz
from ..GeneralDataset.mm_dataset import mgam_Standard_2D
from .meta import CLASS_INDEX_MAP



class AbdomenCT_1K_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class AbdomenCT_1K_Precrop_Npz(AbdomenCT_1K_base, mgam_SemiSup_Precropped_Npz):
    pass


class AbdomenCT_1K_Semi_Mha(AbdomenCT_1K_base, mgam_SemiSup_3D_Mha):
    pass


class AbdomenCT_1K_Sup_2D(AbdomenCT_1K_base, mgam_Standard_2D):
    pass