import pdb
from typing_extensions import Literal, deprecated, Sequence

import cv2
import numpy as np
import SimpleITK as sitk

from mmcv.transforms import BaseTransform
from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size


"""
NOTE
规范化：在进入神经网络之前，
所有预处理的对外特性都应当遵循
[Z,Y,X]或[D,H,W]的维度定义
"""


class BaseLoadBiomedicalData(BaseTransform):
    def _label_remap(self, mask:np.ndarray, label_map:dict):
        mask_copy = mask.copy()
        for old_id, new_id in label_map.items():
            mask[mask_copy == old_id] = new_id
        return mask


class LoadImgFromOpenCV(BaseLoadBiomedicalData):
    """
    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape
    """

    def transform(self, results: dict) -> dict:
        img_path = results["img_path"]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        results["img"] = img
        results["img_shape"] = img.shape[-2:]
        results["ori_shape"] = img.shape[-2:]
        return results


class LoadAnnoFromOpenCV(BaseLoadBiomedicalData):
    """
    Required Keys:

    - seg_map_path

    Modified Keys:

    - gt_seg_map
    - seg_fields
    """

    def transform(self, results: dict) -> dict:
        if "seg_map_path" in results:
            mask_path = results["seg_map_path"]
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"File not found: {mask_path}")
            if results.get("label_map", None) is not None:
                mask = self._label_remap(mask, results["label_map"])

            results["gt_seg_map"] = mask
            results["seg_fields"].append("gt_seg_map")
        return results


class LoadFromMHA(BaseLoadBiomedicalData):
    def __init__(self, resample_spacing=None, resample_size=None):
        assert not ((resample_spacing is not None) and (resample_size is not None))
        self.resample_spacing = resample_spacing
        self.resample_size = resample_size

    def _process_mha(self, mha, field):
        if self.resample_spacing is not None:
            mha = sitk_resample_to_spacing(mha, self.spacing, field)
        if self.resample_size is not None:
            mha = sitk_resample_to_size(mha, self.resample_size, field)
        # mha.GetSize(): [X, Y, Z]
        mha_array = sitk.GetArrayFromImage(mha)  # [Z, Y, X]
        return mha_array


class LoadImageFromMHA(LoadFromMHA):
    """
    Required Keys:

    - img_path

    Modified Keys:

    - img
    - sitk_image
    """

    def transform(self, results):
        img_path = results["img_path"]
        img_mha = sitk.ReadImage(img_path)
        img_mha = sitk.DICOMOrient(img_mha, "LPI")
        img = self._process_mha(img_mha, "image")

        results["img"] = img  # output: [Z, Y, X]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


class LoadMaskFromMHA(LoadFromMHA):
    """
    Required Keys:

    - label_path
    - sitk_image

    Modified Keys:

    - gt_seg_map
    """

    def transform(self, results):
        if "seg_map_path" in results:
            mask_path = results["seg_map_path"]
            mask_mha = sitk.ReadImage(mask_path)
            mask_mha = sitk.DICOMOrient(mask_mha, "LPI")
            mask = self._process_mha(mask_mha, "mask")
            if results.get("label_map", None) is not None:
                mask = self._label_remap(mask, results["label_map"])
            results["gt_seg_map"] = mask  # output: [X, Y, Z]
            results["seg_fields"].append("gt_seg_map")
        return results


class LoadCTPreCroppedSampleFromNpz(BaseLoadBiomedicalData):
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
    DEFAULT_NPZ_FIELDS = ["img", "gt_seg_map"]

    def __init__(self, load_type: VALID_LOAD_FIELD | Sequence[VALID_LOAD_FIELD]):
        self.load_type = load_type if isinstance(load_type, Sequence) else [load_type]
        assert all([load_type in ["img", "anno"] for load_type in self.load_type])

    def transform(self, results):
        assert results["img_path"] == results["seg_map_path"], \
            f"img_path: {results['img_path']}, seg_map_path: {results['seg_map_path']}"
        sample_path = results["img_path"]
        sample = np.load(sample_path)

        if "img" in self.load_type:
            results["img"] = sample[self.DEFAULT_NPZ_FIELDS[0]]
            results["img_shape"] = results["img"].shape[:-1]
            results["ori_shape"] = results["img"].shape[:-1]

        if "anno" in self.load_type:
            gt_seg_map = sample[self.DEFAULT_NPZ_FIELDS[1]]
            # Support mmseg dataset rule
            if results.get("label_map", None) is not None:
                gt_seg_map = self._label_remap(gt_seg_map, results["label_map"])
            results["gt_seg_map"] = gt_seg_map
            results["seg_fields"].append("gt_seg_map")

        return results
