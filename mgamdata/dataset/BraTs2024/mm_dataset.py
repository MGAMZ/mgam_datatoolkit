import os
import pdb
import re
from typing_extensions import Literal, Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from ..base import mgam_BaseSegDataset, mgam_Standard_Precropped_Npz
from .meta import BraTs2024_MODALITIES


class BraTs2024_Dataset(mgam_BaseSegDataset):
    def __init__(self, modality:str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert modality in BraTs2024_MODALITIES
        self.modality = modality
        if self.split == "test":
            self.split = "val"
    
    def sample_iterator(self):
        image_folder = os.path.join(self.data_root, self.split, self.modality)
        label_folder = os.path.join(self.data_root, self.split, "label")
        images = os.listdir(image_folder)
        labels = os.listdir(label_folder)
        
        for series in images:
            img_path = os.path.join(image_folder, series)
            if series in labels:
                ann_path = os.path.join(label_folder, series)
            else:
                ann_path = None
            
            yield img_path, ann_path


class BraTs2024_Precropped(mgam_Standard_Precropped_Npz):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split == "test":
            self.split = "val"

    def _split(self):
        # the train subset contains supervise label,
        # and the val subset does not contain supvervision necessities.
        split_at = "train" if self.mode == "sup" else "val"
        all_series = os.listdir(os.path.join(self.data_root_mha, split_at))
        all_series = sorted(
            all_series, 
            key=lambda x: abs(int(re.search(r"\d+", x).group()))
        )
        np.random.shuffle(all_series)
        total = len(all_series)
        train_end = int(total * self.SPLIT_RATIO[0])
        val_end = train_end + int(total * self.SPLIT_RATIO[1])

        if self.split == "train":
            return all_series[:train_end]
        elif self.split == "val":
            return all_series[train_end:val_end]
        elif self.split == "test":
            return all_series[val_end:]
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")

    def sample_iterator(self):
        for series in self._split():
            one_series_folder = os.path.join(self.data_root, series)
            for file in os.listdir(one_series_folder):
                if file.endswith(".npz"):
                    npz_path = os.path.join(one_series_folder, file)
                    yield npz_path, npz_path


class LoadBraTs2024PreCroppedSample(BaseTransform):
    """
    Required Keys:

    - img_path
    - seg_map_path

    Modified Keys:

    - img
    - gt_seg_map
    - seg_fields
    """
    VALID_LOAD_FIELD = Literal["t1c", "t1n", "t2f", "t2w", "anno"]
    DEFAULT_NPZ_FIELDS = ["t1c", "t1n", "t2f", "t2w", "gt_seg_map"]

    def __init__(self, load_type: VALID_LOAD_FIELD | Sequence[VALID_LOAD_FIELD]):
        self.load_type = load_type if isinstance(load_type, Sequence) else [load_type]
        assert all([load_type in ["t1c", "t1n", "t2f", "t2w", "anno"] for load_type in self.load_type])

    def transform(self, results):
        assert (
            results["img_path"] == results["seg_map_path"]
        ), f"img_path: {results['img_path']}, seg_map_path: {results['seg_map_path']}"
        sample_path = results["img_path"]
        sample = np.load(sample_path)

        img_type = [i for i in self.load_type if i != "anno"]
        if len(img_type) >= 1:
            results["img"] = sample[img_type]
            results["img_shape"] = results["img"].shape[:-1]
            results["ori_shape"] = results["img"].shape[:-1]

        if "anno" in self.load_type:
            gt_seg_map = sample[self.DEFAULT_NPZ_FIELDS[1]]
            # Support mmseg dataset rule
            if results.get("label_map", None) is not None:
                mask_copy = gt_seg_map.copy()
                for old_id, new_id in results["label_map"].items():
                    gt_seg_map[mask_copy == old_id] = new_id
            results["gt_seg_map"] = sample["gt_seg_map"]
            results["seg_fields"].append("gt_seg_map")

        return results