import os
import random
import pdb
from collections.abc import Sequence
from functools import partial
from typing_extensions import Literal, deprecated

import torch
import numpy as np
import cv2
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter

from mmcv.transforms import Resize, BaseTransform
from mmengine.registry import TRANSFORMS


"""
NOTE 
规范化：在进入神经网络之前，
所有预处理的对外特性都应当遵循
[Z,Y,X]或[D,H,W]的维度定义
"""


class AutoPad(BaseTransform):
    def __init__(
        self, 
        size: tuple[int, ...], 
        dim: Literal["1d", "2d", "3d"],
        pad_val: int = 0, 
        pad_label_val: int = 0
    ):
        self.dim = dim
        self.dim_map = {"1d": 1, "2d": 2, "3d": 3}
        if len(size) != self.dim_map[dim]:
            raise ValueError(f"Size tuple length {len(size)} does not match dim {dim}")
        self.size = size
        self.pad_val = pad_val
        self.pad_label_val = pad_label_val

    def _get_pad_params(self, current_shape: tuple) -> tuple[tuple[int, int], ...]:
        pad_params = []
        # 只处理最后n个维度，n由dim决定
        dims_to_pad = self.dim_map[self.dim]
        
        # 确保current_shape维度足够
        if len(current_shape) < dims_to_pad:
            raise ValueError(f"Input shape {current_shape} has fewer dimensions than required {dims_to_pad}")
            
        # 处理不需要padding的前置维度
        for _ in range(len(current_shape) - dims_to_pad):
            pad_params.append((0, 0))
            
        # 处理需要padding的维度
        for target_size, curr_size in zip(self.size, current_shape[-dims_to_pad:]):
            if curr_size >= target_size:
                pad_params.append((0, 0))
            else:
                pad = target_size - curr_size
                pad_1 = pad // 2
                pad_2 = pad - pad_1
                pad_params.append((pad_1, pad_2))
                
        return tuple(pad_params)

    def transform(self, results: dict):
        img = results["img"]
        pad_params = self._get_pad_params(img.shape)
        
        if any(p[0] > 0 or p[1] > 0 for p in pad_params):
            results["img"] = np.pad(
                img,
                pad_params,
                mode="constant",
                constant_values=self.pad_val,
            )
            
            if "gt_seg_map" in results:
                results["gt_seg_map"] = np.pad(
                    results["gt_seg_map"],
                    pad_params,
                    mode="constant",
                    constant_values=self.pad_label_val,
                )

        return results


class CropSlice_Foreground(BaseTransform):
    """
    Required Keys:

    - img
    - gt_seg_map_index
    - gt_seg_map_channel

    Modified Keys:

    - img
    - gt_seg_map_index
    - gt_seg_map_channel
    """

    def __init__(self, num_slices: int, ratio: float = 0.9):
        self.num_slices = num_slices
        self.ratio = ratio  # 一定几率下，本处理才会生效

    def _locate_possible_start_slice_with_non_background(self, mask):
        assert (
            mask.ndim == 3
        ), f"Invalid Mask Shape: Expected [D,H,W], but got {mask.shape}"
        # locate non-background slices
        slices_not_pure_background = np.argwhere(np.any(mask, axis=(1, 2)))
        start_slice, end_slice = (
            slices_not_pure_background.min(),
            slices_not_pure_background.max(),
        )
        non_background_slices = np.arange(start_slice, end_slice, dtype=np.uint32)

        # locate the range of possible start slice,
        # which could ensure the selected slices are not all background
        min_possible_start_slice = max(
            0, non_background_slices[0] - self.num_slices + 1
        )
        max_possible_start_slice = max(
            0, min(non_background_slices[-1], mask.shape[0] - self.num_slices)
        )

        return (min_possible_start_slice, max_possible_start_slice)

    def transform(self, results):
        if np.random.rand(1) > self.ratio:
            return results

        mask = results["gt_seg_map_index"]
        min_start_slice, max_start_slice = (
            self._locate_possible_start_slice_with_non_background(mask)
        )
        selected_slices = np.arange(
            min_start_slice, max_start_slice + self.num_slices - 1
        )

        results["img"] = np.take(results["img"], selected_slices, axis=0)
        results["gt_seg_map_channel"] = np.take(
            results["gt_seg_map_channel"], selected_slices, axis=1
        )
        results["gt_seg_map_index"] = np.take(
            results["gt_seg_map_index"], selected_slices, axis=0
        )
        return results


class WindowSet(BaseTransform):
    """
    Required Keys:

    - img

    Modified Keys:

    - img
    """

    def __init__(self, location, width):
        self.clip_range = (location - width // 2, location + width // 2)
        self.location = location
        self.width = width

    def _window_norm(self, img: np.ndarray):
        img = np.clip(img, self.clip_range[0], self.clip_range[1])  # Window Clip
        img = img - self.clip_range[0]  # HU bias to positive
        img = img / self.width  # Zero-One Normalization
        return img.astype(np.float32)

    def transform(self, results):
        results["img"] = self._window_norm(results["img"])
        return results


class TypeConvert(BaseTransform):
    """
    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map
    """
    def __init__(self, key:str|list[str], dtype:type):
        self.key = key if isinstance(key, list) else [key]
        self.dtype = dtype
    
    def transform(self, results):
        for k in self.key:
            results[k] = results[k].astype(self.dtype)
        return results


class RandomRoll(BaseTransform):
    """
    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map
    """

    def __init__(
        self,
        axis: int | list[int],
        gap: float | list[float],
        erase: bool = False,
        pad_val: int = 0,
        seg_pad_val: int = 0,
    ):
        """
        根据指定的轴进行随机滚动

        :param axis: 指定滚动的轴
        :param gap: 对应轴的最大滚动距离
        :param erase: 是否擦除滚动后的部分
        :param pad_val: 擦除的填充值
        :param seg_pad_val: seg擦除的填充值
        """
        if isinstance(axis, int):
            axis = [axis]
        if isinstance(gap, (int, float)):
            gap = [gap]

        assert len(axis) == len(
            gap
        ), f"axis ({len(axis)}) and gap ({len(gap)}) should have the same length"

        self.axis: list[int] = axis
        self.gap: list[float] = gap
        self.erase = erase
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    @staticmethod
    def _roll(results, gap, axis):
        if "img" in results:
            results["img"] = np.roll(results["img"], shift=gap, axis=axis)
        if "gt_seg_map" in results:
            results["gt_seg_map"] = np.roll(results["gt_seg_map"], shift=gap, axis=axis)
        return results

    def _erase_part(self, results, gap, axis):
        slicer = [slice(None)] * results["img"].ndim
        if gap > 0:
            slicer[axis] = slice(0, gap)
        else:
            slicer[axis] = slice(gap, None)

        if "img" in results:
            results["img"][tuple(slicer)] = self.pad_val
        if "gt_seg_map" in results:
            results["gt_seg_map"][tuple(slicer)] = self.seg_pad_val

        return results

    def transform(self, results):
        for axis, max_gap in zip(self.axis, self.gap):
            gap = random.randint(-max_gap, max_gap)
            results = self._roll(results, gap, axis)
            if self.erase:
                results = self._erase_part(results, gap, axis)
        return results


class InstanceNorm(BaseTransform):
    """
    Required Keys:

    - img

    Modified Keys:

    - img
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def transform(self, results):
        ori_dtype = results["img"].dtype
        img = results["img"]
        img = img - img.min()
        img = img / (img.std() + self.eps)
        results["img"] = img.astype(ori_dtype)
        return results


class ExpandOneHot(BaseTransform):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def transform(self, results):
        mask = results["gt_seg_map"]  # [...]
        # NOTE The ignored index is remapped to the last class.
        if self.ignore_index is not None:
            mask[mask == self.ignore_index] = self.num_classes
        # # eye: Identity Matrix [num_classes+1, num_classes+1]
        mask_channel = np.eye(self.num_classes + 1)[mask]
        mask_channel = np.moveaxis(mask_channel, -1, 0).astype(np.uint8)
        results["gt_seg_map_one_hot"] = mask_channel[:-1]  # [num_classes, ...]
        return results


class GaussianBlur(BaseTransform):
    def __init__(
        self,
        field: list[Literal["image", "label"]],
        kernel_size: int,
        sigma: float,
        amplify: float = 1.0,
    ):
        self.field = field if isinstance(field, list) else [field]
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amplify = amplify
        self.blur = partial(
            cv2.GaussianBlur, ksize=(self.kernel_size, self.kernel_size), sigmaX=sigma
        )

    def transform(self, results: dict):
        if "image" in self.field:
            results["img"] = (self.blur(results["img"]) * self.amplify).astype(np.uint8)
        if "label" in self.field:
            results["gt_seg_map"] = (
                self.blur(results["gt_seg_map"]) * self.amplify
            ).astype(np.float32)
        return results


class GaussianBlur3D(BaseTransform):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def transform(self, results: dict):
        results["img"] = gaussian_filter(results["img"], sigma=self.sigma)
        return results


class RandomGaussianBlur3D(BaseTransform):
    def __init__(self, max_sigma: float, prob: float = 1.0):
        self.sigma = max_sigma
        self.prob = prob

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            sigma = np.random.uniform(0, self.sigma)
            results["img"] = gaussian_filter(results["img"], sigma=sigma)
        return results


class RandomFlip3D(BaseTransform):
    def __init__(self, axis: Literal[0, 1, 2], prob: float = 0.5):
        self.axis = axis
        self.prob = prob

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            if "img" in results:
                results["img"] = np.flip(results["img"], axis=self.axis).copy()
            if "gt_seg_map" in results:
                results["gt_seg_map"] = np.flip(
                    results["gt_seg_map"], axis=self.axis
                ).copy()
        return results


class Resize3D(Resize):
    @staticmethod
    def scale_2D_or_3D(original_shape: list[int], target_shape: list[int]):
        if len(original_shape) == len(target_shape) + 1:
            return [original_shape[0], *target_shape]
        elif len(original_shape) == len(target_shape):
            return target_shape
        else:
            raise ValueError(
                "The dimension of the segmentation map should be equal "
                "to the scale dimension or the scale dimension plus 1, "
                f"but got {original_shape} and {target_shape}"
            )

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        for seg_key in results.get("seg_fields", []):
            if results.get(seg_key, None) is not None:
                scale = self.scale_2D_or_3D(results[seg_key].shape, results["scale"])
                original = torch.from_numpy(results[seg_key])
                results[seg_key] = F.interpolate(
                    original[None, None], size=scale, mode="nearest"
                )[0, 0].numpy()

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        if results.get("img", None) is not None:
            scale = self.scale_2D_or_3D(results["img"].shape, results["scale"])
            original = torch.from_numpy(results["img"].astype(np.float32))
            img = F.interpolate(original[None, None], size=scale, mode="trilinear")

            results["img"] = img[0, 0].numpy().astype(results["img"].dtype)
            results["img_shape"] = img.shape
            results["scale_factor"] = [
                new / ori
                for new, ori in zip(results["img_shape"], results["ori_shape"])
            ]


class RandomCrop3D(BaseTransform):
    """Random crop the 3D volume & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int, int]]):  Expected size after cropping
            with the format of (d, h, w). If set to an integer, then cropping
            depth, width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(
        self,
        crop_size: int | tuple[int, int, int],
        cat_max_ratio: float = 1.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        if isinstance(crop_size, Sequence):
            assert (
                len(crop_size) == 3
            ), f"The expected crop_size containing 3 integers, but got {crop_size}"
        elif isinstance(crop_size, int):
            crop_size = (crop_size, crop_size, crop_size)
        else:
            raise TypeError(f"Unsupported crop size: {crop_size}")

        assert min(crop_size) > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def crop_bbox(self, results: dict, failed_times: int = 0) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped volume.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input volume.

            Returns:
                tuple: Coordinates of the cropped volume.
            """

            margin_d = max(img.shape[0] - self.crop_size[0], 0)
            margin_h = max(img.shape[1] - self.crop_size[1], 0)
            margin_w = max(img.shape[2] - self.crop_size[2], 0)
            offset_d = np.random.randint(0, margin_d + 1)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_d1, crop_d2 = offset_d, offset_d + self.crop_size[0]
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[1]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[2]

            return crop_d1, crop_d2, crop_y1, crop_y2, crop_x1, crop_x2

        img = results["img"]
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.0:
            # Repeat 10 times
            for crop_time in range(10):
                seg_temp = self.crop(results["gt_seg_map"], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if (len(cnt) > 1) and (
                    (np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio
                ):
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input volume.
            crop_bbox (tuple): Coordinates of the cropped volume.

        Returns:
            np.ndarray: The cropped volume.
        """

        crop_d1, crop_d2, crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_d1:crop_d2, crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop volumes, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results["img"]
        crop_bbox = self.crop_bbox(results)

        # crop the volume
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        results["img"] = img
        results["img_shape"] = img.shape[:3]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


class RandomAxis(BaseTransform):
    def __init__(
        self, axis: tuple[Literal[0, 1, 2], Literal[0, 1, 2]], prob: float = 0.5
    ):
        self.axis = axis
        self.prob = prob

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            results["img"] = np.moveaxis(results["img"], self.axis[0], self.axis[1])
            if "gt_seg_map" in results:
                results["gt_seg_map"] = np.moveaxis(
                    results["gt_seg_map"], self.axis[0], self.axis[1]
                )
        return results


class NewAxis(BaseTransform):
    def __init__(self, axis: int):
        self.axis = axis

    def transform(self, results: dict):
        results["img"] = np.expand_dims(results["img"], axis=self.axis)
        return results


class RandomContinuousErase(BaseTransform):
    def __init__(
        self,
        max_size: list[int] | int,
        pad_val: float | int,
        seg_pad_val=0,
        prob: float = 0.5,
    ):
        self.max_size = max_size if isinstance(max_size, (Sequence)) else [max_size]
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.prob = prob

    def _random_area(self, image_size: list[int], area_size:list[int]|None=None
                     ) -> tuple[list[int], list[int]]:
        assert len(image_size) == len(self.max_size)
        dim = len(image_size)
        selected_size = [
                np.random.randint(1, i) for i in self.max_size
            ] if area_size is None else area_size
        
        start_cord = [np.random.randint(0, image_size[i] - selected_size[i]) 
                      for i in range(dim)]
        end_cord = [start_cord[i] + selected_size[i]
                    for i in range(dim)]
        return start_cord, end_cord

    def _erase_area(self, array: np.ndarray, start_cord: list[int], end_cord: list[int]):
        """Erase the information in the selected area, supports any dim"""
        _area = [slice(start_cord[i], end_cord[i]) 
                 for i in range(len(start_cord))]
        array[tuple(_area)] = self.pad_val
        return array

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            cord = self._random_area(results["img"].squeeze().shape)
            results["img"] = self._erase_area(results["img"], cord[0], cord[1])
            if "gt_seg_map" in results:
                results["gt_seg_map"] = self._erase_area(results["gt_seg_map"], cord[0], cord[1])
        return results


class RandomAlter(RandomContinuousErase):
    def _alter_area(self, 
                    array: np.ndarray, 
                    source_area: tuple[list[int], list[int]], 
                    target_area: tuple[list[int], list[int]]):
        """Exchange the information between two local area, supports any dim"""
        source_start, source_end = source_area
        target_start, target_end = target_area
        _source_area = [slice(source_start[i], source_end[i]) 
                        for i in range(len(source_start))]
        _target_area = [slice(target_start[i], target_end[i])
                        for i in range(len(target_start))]
        
        source = array[tuple(_source_area)]
        target = array[tuple(_target_area)]
        array[tuple(_target_area)] = source
        array[tuple(_source_area)] = target
        return array

    def transform(self, results: dict):
        if np.random.rand(1) < self.prob:
            # The location is always randomly determined, 
            # but the size is only determined when source_cord is calculated.
            # Nevertheless, the two area can't alter.
            source_cord = self._random_area(results["img"].squeeze().shape)
            dest_cord = self._random_area(results["img"].squeeze().shape, 
                                          area_size=[source_cord[1][i] - source_cord[0][i]
                                                     for i in range(len(self.max_size))])
            results["img"] = self._alter_area(results["img"], source_cord, dest_cord)
            if "gt_seg_map" in results:
                results["gt_seg_map"] = self._alter_area(results["gt_seg_map"], source_cord, dest_cord)
        return results


class RandomDiscreteErase(BaseTransform):
    def __init__(
        self,
        max_ratio: float,
        pad_val: float | int,
        min_ratio: float = 0.,
        seg_pad_val=0,
        prob: float = 0.5,
    ):
        assert 0 < max_ratio <= 1
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.prob = prob

    def _generate_mask(self, array_shape: tuple, ratio: float) -> np.ndarray:
        total_elements = np.prod(array_shape)
        num_erase = int(total_elements * ratio)
        mask = np.zeros(total_elements, dtype=bool)
        erase_indices = np.random.choice(total_elements, num_erase, replace=False)
        mask[erase_indices] = True
        mask = mask.reshape(array_shape)
        return mask

    def _apply_mask(self, array: np.ndarray, mask: np.ndarray, pad_value) -> np.ndarray:
        if array.ndim > mask.ndim:
            mask = mask[..., None] # channel dim
        array[mask] = pad_value
        return array

    def transform(self, results: dict):
        if np.random.rand() < self.prob:
            erase_ratio = np.random.uniform(self.min_ratio, self.max_ratio)
            img_shape = results["img"].squeeze().shape
            mask = self._generate_mask(img_shape, erase_ratio)
            results["erase_mask"] = mask
            
            results["img"] = self._apply_mask(results["img"], mask, self.pad_val)
            if "gt_seg_map" in results:
                results["gt_seg_map"] = self._apply_mask(results["gt_seg_map"], mask, self.seg_pad_val)
        
        else:
            results["erase_mask"] = np.zeros_like(results["img"].squeeze())
        
        return results


class Identity(BaseTransform):
    def transform(self, results: dict):
        return results


class Resample(BaseTransform):
    def __init__(self, size: list[float], mode: str = "bilinear", field: str = "img"):
        self.size = size
        self.mode = mode
        self.field = field
    
    def transform(self, results: dict):
        results[self.field] = F.interpolate(results[self.field][None, None], size=self.size, mode=self.mode).squeeze()
        return results


class device_to(BaseTransform):
    def __init__(self, key:str|list[str], device:str, non_blocking:bool=False):
        self.key = key if isinstance(key, list) else [key]
        self.device = torch.device(device)
        self.non_blocking = non_blocking
        
    def transform(self, results: dict):
        for key in self.key:
            d = results[key]
            if isinstance(d, torch.Tensor):
                results[key] = d.to(self.device, non_blocking=self.non_blocking)
            elif isinstance(d, np.ndarray):
                results[key] = torch.from_numpy(d).to(self.device, non_blocking=self.non_blocking)
            else:
                raise ValueError(f"Unsupported type {type(d)}")
        return results


class SampleAugment(BaseTransform):
    """
    NOTE
    The reason to do SampleWiseInTimeAugment is the time comsumption
    for IO of an entire sample is too expensive, so it's better
    to augment the sample in time, thus accquiring multiple trainable sub-samples.
    """
    def __init__(self, num_samples:int, pipeline: list[dict]):
        self.num_samples = num_samples
        self.transforms = [TRANSFORMS.build(t) for t in pipeline]
    
    def get_one_sample(self, results: dict):
        for t in self.transforms:
            results = t(results)
        return results
    
    def transform(self, results: dict):
        samples = []
        for _ in range(self.num_samples):
            samples.append(self.get_one_sample(results.copy()))
        return samples

