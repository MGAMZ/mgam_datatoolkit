import os
import pdb
import json
from typing_extensions import Literal, Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from ..base import mgam_Standard_Patched_Npz
from .meta import CLASS_INDEX_MAP


class RoseThyroidCount_base:
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))


class RoseThyroidCount_Precrop_Npz(RoseThyroidCount_base, mgam_Standard_Patched_Npz):
    TEST_SLIDE_UID = ['fd808134e5f32fb1eed8b74afefdf8205bfa1503',
                      'ad935fb82375b9c273765a20f71d9be2c9f60dfe',
                      '41e0bde3dced7b154e098100e9a8a368f03c07c4',
                      '4980726489a59752a823681c2bfeb4bf25e416b6',
                      'ae6509368ead1d0352ccbe57d9b96468c25d94c1']
    SPLIT_RATIO = None

    def _split(self):
        all_series = [i
                      for i in os.listdir(self.data_root)
                      if os.path.isdir(os.path.join(self.data_root, i))]
        assert all([slide in all_series for slide in self.TEST_SLIDE_UID]), f"Missing Test Slide {self.TEST_SLIDE_UID}."
        for slide in self.TEST_SLIDE_UID:
            all_series.remove(slide)

        if self.split == "test" or self.split == "val":
            return self.TEST_SLIDE_UID
        elif self.split == "train":
            return all_series
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")


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
                points = str(sample["point_map"])
                if len(points) > 0:
                    results["points"] = np.array(json.loads(str(sample["point_map"])))
                else:
                    results["points"] = np.array([])
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
