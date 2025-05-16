import os
import pdb
import json
from typing_extensions import Literal, Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from ..base import mgam_BaseSegDataset
from .meta import CLASS_INDEX_MAP


class RoseThyroidCount_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class RoseThyroidCount_Precrop_Npz(RoseThyroidCount_base, mgam_BaseSegDataset):
    SPLIT_RATIO = [0.7, 0.3]

    def _split(self):
        with open(os.path.join(self.data_root, "slide_stats.json"), 'r') as f:
            slide_meta = json.load(f)['slide_details']
        slide_ids = [slide_id for slide_id in list(slide_meta.keys()) 
                     if os.path.exists(os.path.join(self.data_root, slide_id))]
        
        if self.split == 'train':
            return slide_ids[:int(len(slide_ids) * self.SPLIT_RATIO[0])]
        elif self.split == 'val' or self.split == 'test':
            return slide_ids[int(len(slide_ids) * self.SPLIT_RATIO[0]):]
        elif self.split == 'all':
            return slide_ids
        else:
            raise ValueError(f"Invalid split: {self.split}. Expected one of ['train', 'val', 'test']")
    
    def sample_iterator(self):
        for series in self._split():
            series_folder: str = os.path.join(self.data_root, series)
            for sample in os.listdir(series_folder):
                if sample.endswith(".npz"):
                    yield (
                        os.path.join(series_folder, sample),
                        os.path.join(series_folder, sample),
                    )


class LoadRoseThyroidSampleFromNpz(BaseTransform):
    """
    Required Keys:

    - img_path
    - seg_map_path

    Modified Keys:

    - img
    - gt_seg_map
    - seg_fields
    """
    VALID_LOAD_FIELD = Literal["img", "anno"]

    def __init__(self, load_type: VALID_LOAD_FIELD | Sequence[VALID_LOAD_FIELD]):
        self.load_type = load_type if isinstance(load_type, Sequence) else [load_type]
        assert all([load_type in ["img", "anno"] for load_type in self.load_type])

    def transform(self, results):
        assert (results["img_path"] == results["seg_map_path"]), \
            f"img_path: {results['img_path']}, seg_map_path: {results['seg_map_path']}"
        sample_path = results["img_path"]
        sample = np.load(sample_path)

        try:
            if "img" in self.load_type:
                results["img"] = sample["img"]
                results["img_shape"] = results["img"].shape[:-1]
                results["ori_shape"] = results["img"].shape[:-1]

            if "anno" in self.load_type:
                results["points"] = sample["point_map"]
                results["gt_label"] = sample["clustered_cls"]
                
            return results
        
        except Exception as e:
            raise FileNotFoundError(f"Error when loading {sample_path}") from e


class GenPointMap(BaseTransform):
    def __init__(self, size:Sequence[int]):
        self.size = size

    def _gen_point_mask(self, points:list[list[int]]):
        point_mask = np.zeros(self.size, dtype=np.uint8)
        if len(points) > 0:
            for point in points:
                x, y, z = point
                x = np.clip(np.round(x, 0), 0, self.size[1]-1).astype(int)
                y = np.clip(np.round(y, 0), 0, self.size[0]-1).astype(int)
                point_mask[y, x] += 1
        return point_mask

    def transform(self, results:dict):
        results["gt_seg_map"] = self._gen_point_mask(results["points"])
        results["seg_fields"].append("gt_seg_map")
        return results
