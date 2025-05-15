import os
import pdb
import re
import logging
from typing_extensions import deprecated
from pathlib import Path

import pandas as pd

from mmengine.logging import print_log, MMLogger

from . import CLASS_MAP, TEST_SERIES_UIDS, CLASS_MAP_AFTER_POSTSEG
from ..base import (mgam_SemiSup_Precropped_Npz, mgam_SemiSup_3D_Mha, mgam_SeriesPatched_Structure, 
                    mgam_SeriesVolume, mgam_2D_MhaVolumeSlices)


class Sarcopenia_base(mgam_SeriesVolume):
    METAINFO = dict(classes=list(CLASS_MAP.values()))

    def __init__(self, L3_anno_xlsx:str|None=None, ensure_L3_anno=None, *args, **kwargs):
        self.L3_anno_xlsx = L3_anno_xlsx
        self.ensure_L3_anno = ensure_L3_anno if (ensure_L3_anno is not None) else (L3_anno_xlsx is not None)
        self.L3_anno = pd.read_excel(L3_anno_xlsx, usecols=['序列编号', 'L3节段起始层数', 'L3节段终止层数', 'L3节段椎弓根层面层数']) \
                       if L3_anno_xlsx is not None else None
        super().__init__(*args, **kwargs)

    def _split(self):
        # Indexing `SeriesUIDs` according to original mha files.
        split_at = "label" if self.mode == "sup" else "image"
        SeriesUIDs = [
            file.replace(".mha", "")
            for file in os.listdir(os.path.join(self.data_root_mha, split_at))
            if file.endswith(".mha")
        ]
        
        # Exclude Test Series
        exclusion_count = []
        for i, SeriesUID in enumerate(SeriesUIDs):
            if SeriesUID in TEST_SERIES_UIDS:
                exclusion_count.append(i)
                continue
        print_log(f"Excluding project TEST series ({len(exclusion_count)} / {len(SeriesUIDs)}) from {self.split} set.", 
                  MMLogger.get_current_instance(),
                  logging.INFO)
        for i in sorted(exclusion_count, reverse=True):
            SeriesUIDs.pop(i)
        
        # Split and Return
        SeriesUIDs = sorted(SeriesUIDs, key=lambda x: abs(int(re.search(r"\d+", x).group())))
        train_end = int(len(SeriesUIDs) * self.SPLIT_RATIO[0])
        val_end = train_end + int(len(SeriesUIDs) * self.SPLIT_RATIO[1])
        if self.split == "train":
            return SeriesUIDs[:train_end]
        elif self.split == "val":
            return SeriesUIDs[train_end:val_end]
        elif self.split == "test":
            return SeriesUIDs[val_end:]
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")

    def load_data_list(self):
        data_list = super().load_data_list()
        if self.L3_anno is None:
            return data_list
        
        # Add L3 annotation to each sample
        print_log(f"L3 Annotation xlsx file available, adding them into data samples.", MMLogger.get_current_instance())
        to_be_deprecated = []
        for i, data in enumerate(data_list):
            seriesUID = Path(data['img_path']).stem
            L3_anno = self.L3_anno[self.L3_anno['序列编号'] == seriesUID]
            
            if len(L3_anno) == 0:
                if self.ensure_L3_anno is True:
                    print_log(f"无法找到L3标注，由于强制指定需要标注，样本被抛弃: {seriesUID}.", MMLogger.get_current_instance(), logging.WARNING)
                    to_be_deprecated.append(i)
                else:
                    print_log(f"无法找到L3标注，但未抛弃样本: {seriesUID}.", MMLogger.get_current_instance(), logging.INFO)
                    continue
            else:
                # 可能在多个任务集中会对同一个SeriesUID进行标注，仅取最后一个，也即最新的标注。
                data['L3_anno'] = L3_anno[['L3节段起始层数', 'L3节段椎弓根层面层数', 'L3节段终止层数']].iloc[-1].values
        
        # Remove deprecated samples
        if len(to_be_deprecated) > 0:
            for i in sorted(to_be_deprecated, reverse=True):
                data_list.pop(i)
        
        return data_list


class Sarcopenia_Precrop_Npz(Sarcopenia_base, mgam_SemiSup_Precropped_Npz):
    ...

class Sarcopenia_2D_Tiff(Sarcopenia_base, mgam_2D_MhaVolumeSlices):
    ...

class Sarcopenia_Mha(Sarcopenia_base, mgam_SemiSup_3D_Mha):
    ...


class Sarcopenia_base_V2(Sarcopenia_base):
    METAINFO = dict(classes=list(CLASS_MAP_AFTER_POSTSEG.values()))

class Sarcopenia_Precrop_Npz_V2(Sarcopenia_base_V2, mgam_SemiSup_Precropped_Npz):
    ...

class Sarcopenia_2D_Tiff_V2(Sarcopenia_base_V2, mgam_2D_MhaVolumeSlices):
    ...


# Update 250513
class Sarcopenia_Patch_V2(Sarcopenia_base_V2, mgam_SeriesPatched_Structure):
    ...

class Sarcopenia_Mha_V2(Sarcopenia_base_V2, mgam_SemiSup_3D_Mha):
    ...
