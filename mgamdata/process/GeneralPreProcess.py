import os
import random
import pdb
from numbers import Number
from collections.abc import Sequence
from functools import partial
import warnings
from colorama import Fore, Style
from typing_extensions import Literal

import torch
import numpy as np
import cv2
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation as R

from mmcv.transforms import Resize, BaseTransform
from mmengine.registry import TRANSFORMS


"""
NOTE 
规范化：在进入神经网络之前，
所有预处理的对外特性都应当遵循
[Z,Y,X]或[D,H,W]的维度定义
"""


def SetWindow(array:np.ndarray|torch.Tensor, window_width:int, window_level:int):
    window_left = window_level - window_width // 2
    window_right = window_level + window_width // 2
    if isinstance(array, np.ndarray):
        array = np.clip(array, window_left, window_right)
    elif isinstance(array, torch.Tensor):
        array = torch.clamp(array, window_left, window_right)
    else:
        raise TypeError(f"Unsupported type {type(array)}. Expected np.ndarray or torch.Tensor.")
    array = (array - window_left) / window_width
    return array # range: [0, 1]

class AutoPad(BaseTransform):
    def __init__(
        self, 
        size: tuple[int, ...], 
        dim: Literal["1d", "2d", "3d"],
        pad_val: int = 0, 
        pad_label_val: int = 0,
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

    def __init__(self, level, width):
        self.clip_range = (level - width // 2, level + width // 2)
        self.level = level
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
    def __init__(self, key:str|list[str], dtype:str):
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
        inplace: bool = False
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.inplace = inplace

    def transform(self, results):
        mask = results["gt_seg_map"]  # [...]
        # NOTE The ignored index is remapped to the last class.
        if self.ignore_index is not None:
            mask[mask == self.ignore_index] = self.num_classes
        # eye: Identity Matrix [num_classes+1, num_classes+1]
        mask_channel = np.eye(self.num_classes + 1)[mask]
        mask_channel = np.moveaxis(mask_channel, -1, 0).astype(np.uint8)
        
        if self.inplace:
            results["gt_seg_map"] = mask_channel[:-1]
        else:
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
            cv2.GaussianBlur, 
            ksize=(self.kernel_size, self.kernel_size), 
            sigmaX=sigma)

    def transform(self, results: dict):
        if "image" in self.field and "img" in results:
            results["img"] = (self.blur(results["img"]) * self.amplify)
        if "label" in self.field and "gt_seg_map" in results:
            results["gt_seg_map"] = (self.blur(results["gt_seg_map"].astype(np.float32)) * self.amplify)
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

    CROP_RETRY = 32

    def __init__(
        self,
        crop_size: int | tuple[int, int, int],
        cat_max_ratio: float = 1.0,
        std_threshold: float|None = None,
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
        self.std_threshold = std_threshold
        self.ignore_index = ignore_index

    def crop_bbox(self, results: dict) -> tuple:
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

            margin_z = max(img.shape[0] - self.crop_size[0], 0)
            margin_y = max(img.shape[1] - self.crop_size[1], 0)
            margin_x = max(img.shape[2] - self.crop_size[2], 0)
            offset_z = np.random.randint(0, margin_z + 1)
            offset_y = np.random.randint(0, margin_y + 1)
            offset_x = np.random.randint(0, margin_x + 1)
            crop_z1, crop_z2 = offset_z, offset_z + self.crop_size[0]
            crop_y1, crop_y2 = offset_y, offset_y + self.crop_size[1]
            crop_x1, crop_x2 = offset_x, offset_x + self.crop_size[2]

            return crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2

        img = results["img"]
        ann = results["gt_seg_map"]
        
        # crop the volume
        for _ in range(self.CROP_RETRY):
            crop_bbox = generate_crop_bbox(img)
            ccm_check_ = None
            std_check_ = None
            
            # crop check: category max ratio
            if self.cat_max_ratio is not None and self.cat_max_ratio < 1.0:
                seg_temp = self.crop(ann, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if (len(cnt) <= 1) or ((np.max(cnt) / np.sum(cnt)) > self.cat_max_ratio):
                    ccm_check_ = np.max(cnt) / np.sum(cnt)
                    continue
            
            # crop check: std threshold
            if self.std_threshold is not None:
                img_temp = self.crop(img, crop_bbox)
                if img_temp.std() < self.std_threshold:
                    std_check_ = img_temp.std()
                    continue
            
            # when pass all check
            return crop_bbox
        
        else:
            warnings.warn(Fore.YELLOW + \
                          f"Cannot find a valid crop bbox after {self.CROP_RETRY+1} trials. " + \
                          f"Last check result: ccm_check={ccm_check_}, std_check={std_check_}." + \
                          Style.RESET_ALL)
            return None
        
    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input volume.
            crop_bbox (tuple): Coordinates of the cropped volume.

        Returns:
            np.ndarray: The cropped volume.
        """

        crop_z1, crop_z2, crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2, ...]
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
    def __init__(self, axis: int, keys:list[str]=['img']):
        self.axis = axis
        self.keys = keys

    def transform(self, results: dict):
        for k in self.keys:
            results[k] = np.expand_dims(results[k], axis=self.axis)
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
    """
    Args:
        max_ratio (float): The maximum ratio of the erased area.
        keys_pad_vals (Sequence[tuple[str, Number]]): The keys and values to be padded.
        min_ratio (float): The minimum ratio of the erased area.
        prob (float): The probability of performing this transformation.
    
    Modified Keys: 
        Specified by `keys_pad_vals`
    
    Added Keys:
        ori_img (np.ndarray): The original image before erasing.
        erase_mask (np.ndarray): The mask of the erased area.
    """
    
    def __init__(
        self,
        max_ratio: float,
        keys_pad_vals: Sequence[tuple[str, Number]],
        min_ratio: float = 0.,
        prob: float = 0.5,
    ):
        assert 0 < max_ratio <= 1
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.prob = prob
        self.keys_pad_vals = keys_pad_vals

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
        results["ori_img"] = results["img"].copy()
        results["erase_mask"] = np.zeros_like(results["img"].squeeze())
        
        if np.random.rand() < self.prob:
            erase_ratio = np.random.uniform(self.min_ratio, self.max_ratio)
            img_shape = results["img"].squeeze().shape
            mask = self._generate_mask(img_shape, erase_ratio)
            
            results["erase_mask"] = mask
            for key, pad_val in self.keys_pad_vals:
                results[key] = self._apply_mask(results[key], mask, pad_val)
        
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


class RandomRotate3D(BaseTransform):
    def __init__(self,
                 degree: float,
                 prob: float = 1.0,
                 interp_order: int = 0,
                 pad_val: float = -4096,
                 resample_prefilter: bool = False,
                 crop_to_valid_region: bool = True,
                 keys: list[str] = ["img", "gt_seg_map"],
    ):
        self.degree = degree
        self.prob = prob
        self.interp_order = interp_order
        self.pad_val = pad_val
        self.resample_prefilter = resample_prefilter
        self.crop_to_valid_region = crop_to_valid_region
        self.keys = keys
        # 预计算最大旋转角的余弦值
        self.cos_theta = np.cos(np.deg2rad(degree))

    def _sample_rotation_matrix(self):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(-self.degree, self.degree)
        return R.from_rotvec(np.deg2rad(angle) * axis).as_matrix()

    def _rotate_volume(self, array: np.ndarray, rot: np.ndarray):
        z, y, x = array.shape
        center = np.array([z/2, y/2, x/2])

        dz, dy, dx = np.indices((z, y, x))
        coords = np.stack([dz, dy, dx], axis=0).reshape(3, -1).astype(np.float32)

        coords_centered = coords.T - center
        coords_rotated = (rot @ coords_centered.T).T + center

        rotated = map_coordinates(
            array,
            [coords_rotated[:, 0], coords_rotated[:, 1], coords_rotated[:, 2]],
            order=self.interp_order,
            mode="constant",
            cval=self.pad_val,
            prefilter=self.resample_prefilter,
        ).reshape(z, y, x)
        return rotated

    def _find_valid_bounds(self, volume: np.ndarray):
        """使用递归缩小的方式找出有效区域
        
        Args:
            volume: 旋转后的体积数据
            pad_val: padding值
        
        Returns:
            (zmin,zmax,ymin,ymax,xmin,xmax): 有效区域的边界
        """
        shape = np.array(volume.shape)
        center = shape // 2
        
        def check_region(size_ratio):
            """检查给定比例下的区域是否有效"""
            # 计算当前size
            current_size = (shape * size_ratio).astype(int)
            half_size = current_size // 2
            
            # 计算边界
            mins = center - half_size
            maxs = center + half_size
            
            # 提取区域
            region = volume[mins[0]:maxs[0],
                        mins[1]:maxs[1],
                        mins[2]:maxs[2]]
            
            # 检查是否包含pad_val
            return not np.any(region == self.pad_val), (mins, maxs)
        
        # 二分查找最大有效比例
        left, right = 0.0, 1.0
        best_bounds = None
        
        while right - left > 0.01:  # 精度阈值
            mid = (left + right) / 2
            is_valid, bounds = check_region(mid)
            
            # 区域值有效，更新左边界，减小裁切比例
            if is_valid:
                left = mid
                best_bounds = bounds
            # 区域中包含pad_val，更新右边界，增大裁切比例
            else:
                right = mid
        
        if best_bounds is None:
            raise ValueError("No valid region found")
            
        mins, maxs = best_bounds
        return (mins[0], maxs[0]-1,
                mins[1], maxs[1]-1,
                mins[2], maxs[2]-1)

    def _center_crop(self, array: np.ndarray, bounds):
        zmin, zmax, ymin, ymax, xmin, xmax = bounds
        return array[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

    def transform(self, results):
        if np.random.rand() < self.prob:
            rot = self._sample_rotation_matrix()
            
            for key in self.keys:
                # 旋转体积
                rotated = self._rotate_volume(results[key], rot)
                # 找出有效区域
                bounds = self._find_valid_bounds(rotated)
                # 裁剪到有效区域
                if self.crop_to_valid_region:
                    results[key] = self._center_crop(rotated, bounds)
                else:
                    results[key] = rotated
        
        return results


class CenterCrop3D(BaseTransform):
    def __init__(
        self, 
        size: list[int], 
        keys: list[str] = ["img", "gt_seg_map"]
    ):
        self.size = size
        self.keys = keys
    
    def transform(self, results):
        for key in self.keys:
            shape = results[key].shape
            center = np.array(shape) // 2
            half_size = np.array(self.size) // 2
            mins = center - half_size
            maxs = center + half_size
            results[key] = results[key][mins[0]:maxs[0],
                                        mins[1]:maxs[1],
                                        mins[2]:maxs[2]]
        
        if "img_shape" in results:
            results["img_shape"] = self.size
        
        return results
