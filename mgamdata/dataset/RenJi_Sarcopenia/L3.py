from typing import List

import numpy as np
import pandas as pd

from pathlib import Path
from .meta import L3_XLSX_PATH


def find_L3_slices(seriesUIDs: List[str]):
    L3_df = pd.read_excel(L3_XLSX_PATH)
    L3_slicess = [L3_df.loc[L3_df['序列编号'] == seriesUID, 
                            ['L3节段起始层数', 'L3节段终止层数']
                        ].values[0].astype(np.int16)
                  for seriesUID in seriesUIDs]
    return L3_slicess