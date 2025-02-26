from ..GeneralDataset.mm_dataset import mgam_Standard_2D_png

class GastricCancer(mgam_Standard_2D_png):
    METAINFO = dict(classes=["Normal", "Cancer"])