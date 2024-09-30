import os
import pdb
from os import path as osp
from pprint import pprint
from typing import List, Mapping, Tuple, Dict, Union, Sequence
from typing_extensions import deprecated
from tqdm import tqdm
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

import mmcv
import mmengine
from mmengine.logging import print_log, MMLogger
from mmengine.runner import Runner
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.engine.hooks import SegVisualizationHook

from ..dataset.RenJi_Sarcopenia.meta import *




class CT_2D_Sarcopenia(BaseSegDataset):
    SPLIT_RATIO = (0.8, 0.05, 0.15)
    METAINFO = dict(
        classes=list(CLASS_MAP_ABBR.values()),
        palette=list(LABEL_COLOR_DICT.values())
    )
    
    def __init__(self, roots:List[str], split, debug, suffix, *args, **kwargs):
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
        
        for series in used_series:
            yield series
    
    def load_data_list(self) -> List[dict]:
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
                 use_folds:Union[int, List[int]], 
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
    def __init__(self, window_width:int=None, window_location:int=None, *args, **kwargs):
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



@deprecated("应当从sitk toolkit中继承或调用，确保一致性")
def MhaResampleToTarget(source_image:sitk.Image,
                        resample_type:str,
                        target_image:sitk.Image|None = None,
                        target_size = None,
                        target_spacing = None):
    
    assert resample_type in ['image', 'mask']
    valid_params_count = sum([target_image is not None,
                              target_size is not None,
                              target_spacing is not None])
    # 检查是否只有一个有效参数
    if valid_params_count != 1:
        raise ValueError("Exactly one of 'target_image', 'target_size', or 'target_spacing' must be provided.")
    
    # 当target_image不为空时，直接对齐TargetImage
    if target_image is not None:
        target_size = target_image.GetSize()
        target_spacing = target_image.GetSpacing()
    
    # 当target_size或target_spacing不为空时，
    # 根据其中一者计算另一者的参数。
    else:
        if target_size is not None:
            original_size = source_image.GetSize()
            original_spacing = source_image.GetSpacing()
            target_spacing = [original_spacing[i] * original_size[i] / target_size[i] 
                              for i in range(3)]
            
        elif target_spacing is not None:
            original_size = source_image.GetSize()
            original_spacing = source_image.GetSpacing()
            target_size = [int(original_size[i] * original_spacing[i] / target_spacing[i]) 
                           for i in range(3)]
    
    resampled_image = sitk.Resample(
        image1=source_image,
        size=target_size,
        interpolator=sitk.sitkLinear if resample_type == 'image' else sitk.sitkNearestNeighbor,
        outputSpacing=target_spacing,
        outputPixelType=sitk.sitkInt16 if resample_type == 'image' else sitk.sitkUInt8,
        outputOrigin=source_image.GetOrigin(),
        outputDirection=source_image.GetDirection(),
        transform=sitk.Transform(),
    )

    return resampled_image



