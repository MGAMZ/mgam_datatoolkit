import os
import pdb
import logging
from os import path as osp
from re import L
from regex import F
from typing_extensions import deprecated
from tqdm import tqdm
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd

import mmcv
import mmengine
from mmengine.logging import print_log, MMLogger
from mmengine.runner import Runner
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.engine.hooks import SegVisualizationHook

from . import (
    CLASS_MAP, CLASS_MAP_ABBR, LABEL_COLOR_DICT,
    HUANGSHAN_HOSPITAL_SERIES_UIDS,
    RENJI_HOSPITAL_DUPLICATED_SERIES_UIDS,
    ZHEJIANG_HOSPITAL_SERIES_UIDS,
    WENZHOU_HOSPITAL_SERIES_UIDS, TEST_SERIES_UIDS
)
from ..base import mgam_SemiSup_Precropped_Npz, mgam_SemiSup_3D_Mha, mgam_BaseSegDataset



class CT_2D_Sarcopenia(BaseSegDataset):
    SPLIT_RATIO = (0.8, 0.05, 0.15)
    METAINFO = dict(
        classes=list(CLASS_MAP_ABBR.values()),
        palette=list(LABEL_COLOR_DICT.values())
    )
    
    def __init__(self, roots:list[str], split, debug, suffix, *args, **kwargs):
        self.roots = roots
        self.split = split
        self.debug = debug
        self.suffix = suffix
        super().__init__(
            img_suffix=suffix, 
            seg_map_suffix=suffix, 
            reduce_zero_label=False, 
            *args, **kwargs)
    
    def _indexing(self):
        # Index
        seriesUIDs = []
        available_series = []
        for root in self.roots:
            label_folder = osp.join(root, 'label')
            image_folder = osp.join(root, 'image')
            if not osp.exists(label_folder) or not osp.exists(image_folder):
                raise FileNotFoundError(f"Invalid Data Root: {root}")
            
            for mha_file_name in os.listdir(label_folder):
                seriesUID = Path(mha_file_name).stem
                image_serial = osp.join(image_folder, mha_file_name)
                label_serial = osp.join(label_folder, mha_file_name)
                # 不存在有效扫描或者已经索引过的时候，跳过。
                if (not osp.exists(image_serial)) or (seriesUID in seriesUIDs):
                    continue
                available_series.append((image_serial, label_serial))
                seriesUIDs.append(seriesUID)
        
        return sorted(available_series)
    
    def _split(self):
        available_series = self._indexing()
        
        # Split
        split_border = (int(len(available_series) * self.SPLIT_RATIO[0]),
                        int(len(available_series) * (self.SPLIT_RATIO[0] + self.SPLIT_RATIO[1])))
        if self.split == 'train':
            used_series = available_series[:split_border[0]]
        elif self.split == 'val':
            used_series = available_series[split_border[0]:split_border[1]]
        elif self.split == 'test':
            used_series = available_series[split_border[1]:-1]
        else:
            raise TypeError(f"Not supported sub-dataset split: {self.split}")
        
        for series in used_series:
            yield series
    
    def load_data_list(self) -> list[dict]:
        # Attention: case is indexed by mask_root
        data_list = []
        for image_folder, label_folder in self._split():
            for file in os.listdir(label_folder):
                if file.endswith(self.suffix):
                    img_path = osp.join(image_folder, file)
                    label_path = osp.join(label_folder, file)
                    data_info = dict(
                        img_path=img_path,
                        seg_map_path=label_path,
                        label_map=self.METAINFO,
                        reduce_zero_label=False,
                        seg_fields=[])
                    data_list.append(data_info)
        
        if self.debug:
            data_list = data_list[:8]
        print_log(f"{self.split} set sample: {len(data_list)}.", 
                  MMLogger.get_current_instance())
        return data_list


class CT_2D_Sar_CrossFold(CT_2D_Sarcopenia):
    def __init__(self, 
                 use_folds:int|list[int], 
                 total_folds:int = 5, 
                 *args, **kwargs):
        if isinstance(use_folds, int):
            use_folds = [use_folds]
        assert min(use_folds) >= 1 and max(use_folds) <= total_folds
        self.use_folds = use_folds
        self.total_folds = total_folds
        super().__init__(split=None, *args, **kwargs)
    
    def _split(self):
        available_series = self._indexing()
        borders = np.linspace(0, len(available_series), self.total_folds + 1, dtype=np.uint32)
        
        used_series = []
        for fold_id in self.use_folds:
            start, end = borders[fold_id-1], borders[fold_id]
            used_series += available_series[start:end]
        
        for series in used_series:
            yield series


class CT_2D_Sar_CrossFold_OnlyRenJiData(CT_2D_Sar_CrossFold):
    def _indexing(self):
        series = super()._indexing()
        renji_series = []
        for serial in series:
            if Path(serial[0]).name in HUANGSHAN_HOSPITAL_SERIES_UIDS:
                print_log(f"Skip Huangshan Hospital Series: {serial[0]}", MMLogger.get_current_instance())
                continue
            if Path(serial[0]).name in RENJI_HOSPITAL_DUPLICATED_SERIES_UIDS:
                print_log(f"Skip Renji Hospital Duplicated Series: {serial[0]}", MMLogger.get_current_instance())
                continue
            if Path(serial[0]).name in ZHEJIANG_HOSPITAL_SERIES_UIDS:
                print_log(f"Skip Zhejiang Hospital Series: {serial[0]}", MMLogger.get_current_instance())
                continue
            if Path(serial[0]).name in WENZHOU_HOSPITAL_SERIES_UIDS:
                print_log(f"Skip Wenzhou Hospital Series: {serial[0]}", MMLogger.get_current_instance())
                continue
            renji_series.append(serial)
        return renji_series


class CT_VisualizationHook(SegVisualizationHook):
    def __init__(self, 
                 window_width:int|None=None, 
                 window_location:int|None=None, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ww = window_width
        self.wl = window_location
    
    
    def _load_original_image(self, img_bytes:bytes) -> np.ndarray:
        img = mmcv.imfrombytes(img_bytes, backend='tifffile')
        if self.ww is not None and self.wl is not None:
            img = np.clip(img, self.wl - self.ww//2, self.wl + self.ww//2)
        img = img - img.min()
        img = (img / img.max() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    
    
    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs) -> None:
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        img_bytes = mmengine.fileio.get(img_path, backend_args=self.backend_args)
        img = self._load_original_image(img_bytes)
        window_name = f'val_{osp.basename(img_path)}'

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                step=total_curr_iter)


    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            window_name = f'test_{osp.basename(img_path)}'

            img_path = data_sample.img_path
            img_bytes = mmengine.fileio.get(img_path, backend_args=self.backend_args)
            img = self._load_original_image(img_bytes)

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index)


class Sarcopenia_base:
    METAINFO = dict(classes=list(CLASS_MAP.values()))

    def __init__(self, L3_anno_xlsx:str|None=None, ensure_L3_anno=None, *args, **kwargs):
        self.L3_anno_xlsx = L3_anno_xlsx
        self.ensure_L3_anno = ensure_L3_anno if (ensure_L3_anno is not None) else (L3_anno_xlsx is not None)
        self.L3_anno = pd.read_excel(L3_anno_xlsx, usecols=['序列编号', 'L3节段起始层数', 'L3节段终止层数', 'L3节段椎弓根层面层数']) \
                       if L3_anno_xlsx is not None else None
        super().__init__(*args, **kwargs)

    def load_data_list(self):
        data_list = mgam_BaseSegDataset.load_data_list(self)
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
    def load_data_list(self):
        data_list = super().load_data_list()
        exclusion_count = 0
        for data in data_list:
            file_name = Path(data['img_path']).name
            dir_name = Path(data['img_path']).parent.name
            if dir_name in TEST_SERIES_UIDS:
                exclusion_count += 1
                continue
        
        print_log(f"Split {self.split} excluded samples for Sarcopenia Product Test: {exclusion_count} out of {len(data_list)}", 
                  MMLogger.get_current_instance())
        return data_list


class Sarcopenia_Mha(Sarcopenia_base, mgam_SemiSup_3D_Mha):
    def load_data_list(self):
        data_list = super().load_data_list()
        exclusion_count = 0
        for data in data_list:
            file_name = Path(data['img_path']).name
            if file_name in TEST_SERIES_UIDS:
                exclusion_count += 1
                continue
        
        print_log(f"Split {self.split} excluded samples for Sarcopenia Product Test: {exclusion_count} out of {len(data_list)}", 
                  MMLogger.get_current_instance())
        return data_list

