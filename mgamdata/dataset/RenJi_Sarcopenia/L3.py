import pdb
import warnings
from typing import List

import numpy as np
import pandas as pd

from ..RenJi_Sarcopenia import L3_XLSX_PATH


def find_L3_slices(seriesUIDs: List[str]):
    L3_df = pd.read_excel(L3_XLSX_PATH)
    L3_slicess = []
    for seriesUID in seriesUIDs:
        series_anno = L3_df.loc[L3_df['序列编号'] == seriesUID]
        series_L3 = series_anno[['L3节段起始层数', 'L3节段终止层数']].values.flatten()
        if np.isnan(series_L3).any():
            L3_slicess.append(None)
        else:
            L3_slicess.append(series_L3.astype(np.uint32))
    return L3_slicess