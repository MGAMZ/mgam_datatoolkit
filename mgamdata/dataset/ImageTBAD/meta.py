import os

DATA_ROOT = os.environ['ImageTBAD_data_root']
DATA_ROOT_3D_MHA = os.path.join(DATA_ROOT, 'original_mha')

CLASS_INDEX_MAP = {
    "background": 0,
    "TL": 1,
    "FL": 2,
    "FLT": 3
}