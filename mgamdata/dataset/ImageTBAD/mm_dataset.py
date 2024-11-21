from ..base import mgam_Standard_3D_Mha, mgam_Standard_Precropped_Npz
from .meta import CLASS_INDEX_MAP



class ImageTBAD_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class TBAD_Precrop_Npz(ImageTBAD_base, mgam_Standard_Precropped_Npz):
    ...


class TBAD_Mha(ImageTBAD_base, mgam_Standard_3D_Mha):
    ...