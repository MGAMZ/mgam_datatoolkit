import pdb
from typing_extensions import deprecated, Sequence

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


class LoadImgFromOpenCV(BaseTransform):
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


class LoadAnnoFromOpenCV(BaseTransform):
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
            if results.get("label_map", None) is not None:
                mask_copy = mask.copy()
                for old_id, new_id in results["label_map"].items():
                    mask[mask_copy == old_id] = new_id

            results["gt_seg_map"] = mask
            results["seg_fields"].append("gt_seg_map")
        return results


class LoadFromMHA(BaseTransform):
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
        mask_path = results["seg_map_path"]
        mask_mha = sitk.ReadImage(mask_path)
        mask = self._process_mha(mask_mha, "mask")
        if results.get("label_map", None) is not None:
            mask_copy = mask.copy()
            for old_id, new_id in results["label_map"].items():
                mask[mask_copy == old_id] = new_id
        results["gt_seg_map"] = mask  # output: [X, Y, Z]
        results["seg_fields"].append("gt_seg_map")
        return results


class LoadSampleFromNpz(BaseTransform):
    """
    Required Keys:

    - img_path
    - seg_map_path

    Modified Keys:

    - img
    - gt_seg_map
    - seg_fields
    """

    def __init__(self, load_type: str | Sequence[str]):
        self.load_type = load_type if isinstance(load_type, Sequence) else [load_type]
        assert all([load_type in ["img", "anno"] for load_type in self.load_type])

    def transform(self, results):
        assert (
            results["img_path"] == results["seg_map_path"]
        ), f"img_path: {results['img_path']}, seg_map_path: {results['seg_map_path']}"
        sample_path = results["img_path"]
        sample = np.load(sample_path)

        if "img" in self.load_type:
            results["img"] = sample["img"]
            results["img_shape"] = results["img"].shape[:-1]
            results["ori_shape"] = results["img"].shape[:-1]

        if "anno" in self.load_type:
            point_mask = sample["heatmap"]
            cluster_cls = sample["clustered"]
            # Support mmseg dataset rule
            if results.get("label_map", None) is not None:
                mask_copy = point_mask.copy()
                for old_id, new_id in results["label_map"].items():
                    point_mask[mask_copy == old_id] = new_id
            results["gt_seg_map"] = point_mask
            results["gt_label"] = cluster_cls
            results["seg_fields"].append("gt_seg_map")

        return results


@deprecated("`PackSegInputs` will perform the same operation.")
class EnsureChannelDim(BaseTransform):
    def transform(self, results):
        # preprocessing on image requires [..., C]
        # the C will be move to the head in `PackSegInputs` transformation.
        if "img" in results:
            if len(results["img"].shape) == 2:
                results["img"] = results["img"][..., None]
        if "gt_seg_map" in results:
            if len(results["gt_seg_map"].shape) == 2:
                results["gt_seg_map"] = results["gt_seg_map"][None, ...]
        return results
