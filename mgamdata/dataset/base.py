import os
import re
import pdb
import json
import logging
from abc import abstractmethod
from collections.abc import Generator, Iterable
from tqdm import tqdm
from typing_extensions import Literal, deprecated

import orjson
import numpy as np

from mmcv.transforms import BaseTransform
from mmengine.registry import DATASETS
from mmengine.logging import print_log, MMLogger
from mmengine.dataset import ConcatDataset, BaseDataset
from mmseg.datasets.basesegdataset import BaseSegDataset


class mgam_BaseSegDataset(BaseSegDataset):
    SPLIT_RATIO = [0.8, 0.05, 0.15]

    def __init__(
        self,
        split: str|None,
        debug: bool = False,
        dataset_name: str | None = None,
        *args, **kwargs,
    ) -> None:
        self.split = split
        self.debug = debug
        assert debug in [True, False]
        self.dataset_name = (dataset_name 
                             if dataset_name is not None 
                             else self.__class__.__name__)
        super().__init__(*args, **kwargs)
        self.data_root: str
        # HACK emergency override for sarcopenia training.
        # assert self.img_suffix == self.seg_map_suffix, \
        #     f"img_suffix {self.img_suffix} and seg_map_suffix {self.seg_map_suffix} should be the same"

    def _update_palette(self) -> list[list[int]]:
        """确保background为RGB全零"""
        new_palette = super()._update_palette()

        if len(self.METAINFO) > 1:
            return [[0, 0, 0]] + new_palette[1:]
        else:
            return new_palette

    @abstractmethod
    def sample_iterator(self) -> Generator[tuple[str, str], None, None] | Iterable[tuple[str, str]]: 
        ...

    def load_data_list(self):
        """
        Sample Required Keys in mmseg:

        - img_path: str, 图像路径
        - seg_map_path: str, 分割标签路径
        - label_map: str, 分割标签的类别映射，默认为空。它是矫正映射，如果map没有问题，则不需要矫正。
        - reduce_zero_label: bool, 是否将分割标签中的0类别映射到-1(255), 默认为False
        - seg_fields: list, 分割标签的字段名, 默认为空列表
        """
        data_list = []
        for image_path, anno_path in self.sample_iterator():
            data_list.append(
                dict(
                    img_path=image_path,
                    seg_map_path=anno_path,
                    label_map=self.label_map,
                    reduce_zero_label=False,
                    seg_fields=[],
                )
            )

        if self.debug:
            print_log(
                f"{self.dataset_name} dataset {self.split} split loaded {len(data_list)} samples, "
                f"DEBUG MODE ENABLED, ONLY 16 SAMPLES ARE USED",
                MMLogger.get_current_instance(),
                logging.WARNING
            )
            return data_list[:16]
        else:
            print_log(
                f"{self.dataset_name} dataset {self.split} split loaded {len(data_list)} samples.",
                MMLogger.get_current_instance(),
            )
            return data_list


class mgam_SeriesVolume(mgam_BaseSegDataset):
    def __init__(self,
                 data_root_mha:str,
                 mode:Literal["semi", "sup"]="sup",
                 *args, **kwargs):
        # `Semi` mode will still include those samples without labels
        # `Sup` mode will exclude those samples without labels
        self.mode = mode
        self.data_root_mha = data_root_mha
        super().__init__(*args, **kwargs)
        self.data_root: str
    
    def _split(self):
        split_at = "label" if self.mode == "sup" else "image"
        all_series = [
            file.replace(".mha", "")
            for file in os.listdir(os.path.join(self.data_root_mha, split_at))
            if file.endswith(".mha")
        ]
        all_series = sorted(all_series, key=lambda x: abs(int(re.search(r"\d+", x).group())))
        train_end = int(len(all_series) * self.SPLIT_RATIO[0])
        val_end = train_end + int(len(all_series) * self.SPLIT_RATIO[1]) + 1

        if self.split == "train":
            return all_series[:train_end]
        elif self.split == "val":
            return all_series[train_end:val_end]
        elif self.split == "test":
            return all_series[val_end:]
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")


class mgam_2D_MhaVolumeSlices(mgam_SeriesVolume):
    def sample_iterator(self) -> Generator[tuple[str, str], None, None]:
        for series in self._split():
            series_folder = os.path.join(self.data_root, 'label' if self.mode=='sup' else 'image', series)
            if not os.path.exists(series_folder):
                print_log(f"{series} not found.\nFullPath: {series_folder}",
                          MMLogger.get_current_instance(),
                          logging.WARN)
                continue
            for sample in os.listdir(series_folder):
                if sample.endswith(self.img_suffix):
                    yield (os.path.join(self.data_root, 'image', series, sample),
                           os.path.join(self.data_root, 'label', series, sample))


class mgam_SemiSup_3D_Mha(mgam_SeriesVolume):
    def sample_iterator(self) -> Generator[tuple[str, str], None, None]:
        for series in self._split():
            image_mha_path = os.path.join(self.data_root, "image", series + ".mha")
            label_mha_path = os.path.join(self.data_root, "label", series + ".mha")
            if not os.path.exists(image_mha_path):
                print_log(f"{series} image mha file not found.\nFullPath: {image_mha_path}",
                          MMLogger.get_current_instance(),
                          logging.WARN)
                continue
            yield (image_mha_path, label_mha_path)


class mgam_SemiSup_Precropped_Npz(mgam_SemiSup_3D_Mha):
    def __init__(self, *args, **kwargs) -> None:
        with open(os.path.join(kwargs["data_root"], "crop_meta.json"), "r") as f:
            self.precrop_meta = json.load(f)
        super().__init__(*args, **kwargs)

    def sample_iterator(self) -> Generator[tuple[str, str], None, None]:
        for series in tqdm(self._split(),
                           desc=f"Indexing {self.split} samples of all series of {self.__class__.__name__}",
                           leave=False,
                           dynamic_ncols=True):
            # Check usability.
            if self.mode == "sup" and series not in self.precrop_meta["anno_available"]:
                continue
            series_folder = os.path.join(self.data_root, series)
            try:
                series_meta = orjson.loads(open(os.path.join(series_folder, "SeriesMeta.json"), "r").read())
            except FileNotFoundError:
                print_log(f"{series} not found.", MMLogger.get_current_instance())
                continue
            
            patch_npz_files = series_meta["class_within_patch"].keys()
            for sample in [os.path.join(series_folder, file) 
                           for file in patch_npz_files]:
                if sample.endswith(".npz"):
                    yield (os.path.join(series_folder, sample),
                           os.path.join(series_folder, sample))


class mgam_SeriesPatched_Structure(mgam_SeriesVolume):
    def __init__(self, *args, **kwargs) -> None:
        with open(os.path.join(kwargs["data_root"], "crop_meta.json"), "r") as f:
            self.precrop_meta = json.load(f)
        super().__init__(*args, **kwargs)

    def sample_iterator(self) -> Generator[tuple[str, str], None, None]:
        for series in tqdm(self._split(),
                           desc=f"Indexing {self.split} samples of all series of {self.__class__.__name__}",
                           leave=False,
                           dynamic_ncols=True):
            # Check usability.
            if self.mode == "sup" and series not in self.precrop_meta["anno_available"]:
                continue
            series_folder = os.path.join(self.data_root, series)
            try:
                with open(os.path.join(series_folder, "SeriesMeta.json"), "r") as f:
                    series_meta = json.load(f)
            except FileNotFoundError:
                print_log(f"{series} not found.", MMLogger.get_current_instance())
                continue
            
            patch_npz_files = series_meta["class_within_patch"].keys()
            for sample in [os.path.join(series_folder, file) 
                           for file in patch_npz_files]:
                yield (os.path.join(series_folder, sample.replace('_label', '_image')),
                       os.path.join(series_folder, sample))


class mgam_concat_dataset(ConcatDataset):
    def __init__(
        self,
        datasets: list[BaseDataset | dict],
        lazy_init: bool = False,
        ignore_keys: str | list[str] | None = None,
    ):
        self.datasets: list[BaseDataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    "elements in datasets sequence should be config or "
                    f"`BaseDataset` instance, but got {type(dataset)}"
                )
        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError(
                "ignore_keys should be a list or str, " f"but got {type(ignore_keys)}"
            )

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # Only use metainfo of first dataset.
        self._metainfo = self.datasets[0].metainfo

        # HACK MGAM: Skip dataset-wise metainfo consistent check

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()
        
        print_log(f"ConcatDataset loaded {len(self)} samples.", MMLogger.get_current_instance())



class unsup_base:
    METAINFO = dict(classes=["background"])


class unsup_base_Precrop_Npz(unsup_base, mgam_SeriesPatched_Structure):
    pass


class unsup_base_Semi_Mha(unsup_base, mgam_SemiSup_3D_Mha):
    pass
