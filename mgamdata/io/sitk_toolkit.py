import os
import os.path as osp
import pdb
import warnings
from glob import glob
from colorama import Style, Fore
from typing_extensions import Literal

import pydicom
import numpy as np
import SimpleITK as sitk



STANDARD_DIRECTION = [1, 0, 0, 0, 1, 0, 0, 0, 1]
STANDARD_ORIGIN = [0, 0, 0]
PIXEL_TYPE = lambda field: sitk.sitkInt16 if field == "image" else sitk.sitkUInt8
INTERPOLATOR = lambda field: sitk.sitkBSpline3 if field == "image" else sitk.sitkNearestNeighbor


def sitk_resample_to_spacing(mha:sitk.Image, 
                             spacing: list[float], 
                             field: Literal["image", "label"]):
    """改进后的重采样方法。

    Args:
        mha (sitk.Image): 输入sitk.Image
        spacing (tuple[float,float,float]): 新的spacing [Z,Y,X]
        field (str, optional): 重采样的对象。 可选'image', 'label', 'mask'.
                               本参数将决定插值方法和数据格式。
        standardize (bool, optional): 是否标准化输出的origin和direction. Defaults to False.

    Returns:
        sitk.Image: 重采样后的sitk.Image
    """
    assert field in ["image", "label"], "field must be one of ['image', 'label']"
    assert len(spacing) == 3, f"Spacing must be a 3-tuple, got {spacing}"
    
    # 计算重采样后的Spacing
    spacing = spacing[::-1]
    original_spacing = mha.GetSpacing()
    for i in range(3):
        if spacing[i] == -1:
            spacing[i] = original_spacing[i]
        else:
            assert spacing[i] > 0, f"Spacing must be positive or -1 (Not Changed), but got {spacing}"
    
    if original_spacing == spacing:
        return mha
    
    original_size = mha.GetSize()
    spacing_ratio = [original_spacing[i] / spacing[i] for i in range(3)]
    resampled_size = [int(original_size[i] * spacing_ratio[i]) for i in range(3)]
    
    # 执行重采样
    try:
        return sitk.Resample(
            image1=mha,
            size=resampled_size,  # type:ignore
            interpolator=INTERPOLATOR(field),
            outputSpacing=spacing,
            outputPixelType=PIXEL_TYPE(field),
            outputOrigin=mha.GetOrigin(),
            outputDirection=mha.GetDirection(),
            transform=sitk.Transform(),
        )
    except Exception as e:
        return {
            "error": f"Failed to resample image: {e}",
            "original_size": original_size,
            "original_spacing": original_spacing,
            "spacing": spacing,
            "resampled_size": resampled_size,
            "mha": str(mha),
            "field": field,
        }


def sitk_resample_to_image(
    image: sitk.Image,
    reference_image: sitk.Image,
    field: Literal["image", "label"],
    default_value=0.0,
):
    """重采样一个sitk.Image，对齐到另一个sitk.Image。

    Args:
        image (sitk.Image): 输入的sitk.Image
        reference_image (sitk.Image): 对齐目标
        field (str, optional): 重采样的对象。 可选'image', 'label', 'mask'.
                               本参数将决定插值方法和数据格式.
        default_value (float, optional): 重采样时填充值. Defaults to 0.

    Returns:
        sitk.Image: 重采样后的sitk.Image
    """
    return sitk.Resample(
        image1=image,
        size=reference_image.GetSize(),
        interpolator=INTERPOLATOR(field),
        outputSpacing=reference_image.GetSpacing(),
        outputPixelType=PIXEL_TYPE(field),
        outputOrigin=reference_image.GetOrigin(),
        outputDirection=reference_image.GetDirection(),
        defaultPixelValue=default_value,
    )


def sitk_resample_to_size(
    image,
    new_size: list[float],
    field: Literal["image", "label"],
):
    """
    Args:
        image (sitk.Image): 输入的sitk.Image
        new_size (list[float]): 新的大小
        field (str, optional): 重采样的对象。 可选'image', 'label', 'mask'.
                               本参数将决定插值方法和数据格式.
        standardize (bool, optional): 是否标准化输出的origin和direction. Defaults to False.

    Returns:
        sitk.Image: 重采样后的sitk.Image
    """
    assert len(new_size) == 3, f"Size must be a 3-tuple, got {new_size}"
    
    new_size = new_size[::-1]
    original_size = image.GetSize()
    for i in range(3):
        if new_size[i] == -1:
            new_size[i] = original_size[i]
        else:
            assert new_size[i] > 0, f"Size must be positive or -1 (Not Changed), but got {new_size}"
    
    if new_size == original_size:
        return image
    
    original_spacing = image.GetSpacing()
    new_spacing = np.divide(original_spacing, np.divide(new_size, original_size))
    
    return sitk.Resample(
        image1=image,
        size=new_size,
        interpolator=INTERPOLATOR(field),
        outputSpacing=new_spacing,
        outputPixelType=PIXEL_TYPE(field),
        outputOrigin=image.GetOrigin(),
        outputDirection=image.GetDirection(),
        transform=sitk.Transform(),
    )


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.0):
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float32).T * default_value, isVector=False)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def nii_to_sitk(
    nii_path: str,
    field: Literal["image", "label"],
    value_offset: int | float | None = None,
) -> sitk.Image:
    try:
        sitk_img = sitk.ReadImage(nii_path, outputPixelType=sitk.sitkInt16 if field == "image" else sitk.sitkUInt8)
        if value_offset is not None:
            sitk_img_new = sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk_img) + value_offset)
            sitk_img_new.CopyInformation(sitk_img)
            sitk_img = sitk_img_new
    
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file: {nii_path}.") from e

    return sitk_img


def LoadDcmAsSitkImage(dcm_paths, read_workers=8):
    dcms = []

    for dcm_path in dcm_paths:
        ds = pydicom.dcmread(dcm_path, force=True)
        if (0x20, 0x32) not in ds:  # (0020, 0032) Image Position (Patient)
            warnings.warn(
                Fore.YELLOW
                + f"ImagePosition Missing, Deprecating: {dcm_paths}"
                + Style.RESET_ALL
            )
            return False
        dcms.append((dcm_path, ds[0x20, 0x32].value[-1]))
    else:
        dcms = sorted(dcms, key=lambda x: x[1], reverse=False)

    sorted_dcm_paths = [dcm[0] for dcm in dcms]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_dcm_paths)
    reader.SetNumberOfWorkUnits(read_workers)
    sitk_image: sitk.Image = reader.Execute()

    return sitk_image


def LoadDcmAsSitkImage_EngineeringOrder(
    dcm_case_path, spacing, sort_by_distance=True
) -> tuple[
    sitk.Image,
    tuple[float, float, float] | None,
    tuple[int, int, int] | None,
    tuple[int, int, int] | None,
]:
    # Spacing: [D, H, W]

    class DcmInfo(object):
        def __init__(
            self,
            dcm_path,
            series_instance_uid,
            acquisition_number,
            sop_instance_uid,
            instance_number,
            image_orientation_patient,
            image_position_patient,
        ):
            super(DcmInfo, self).__init__()

            self.dcm_path = dcm_path
            self.series_instance_uid = series_instance_uid
            self.acquisition_number = acquisition_number
            self.sop_instance_uid = sop_instance_uid
            self.instance_number = instance_number
            self.image_orientation_patient = image_orientation_patient
            self.image_position_patient = image_position_patient

            self.slice_distance = self._cal_distance()

        def _cal_distance(self):
            normal = [
                self.image_orientation_patient[1] * self.image_orientation_patient[5]
                - self.image_orientation_patient[2] * self.image_orientation_patient[4],
                self.image_orientation_patient[2] * self.image_orientation_patient[3]
                - self.image_orientation_patient[0] * self.image_orientation_patient[5],
                self.image_orientation_patient[0] * self.image_orientation_patient[4]
                - self.image_orientation_patient[1] * self.image_orientation_patient[3],
            ]

            distance = 0
            for i in range(3):
                distance += normal[i] * self.image_position_patient[i]
            return distance

    def is_sop_instance_uid_exist(dcm_info, dcm_infos):
        for item in dcm_infos:
            if dcm_info.sop_instance_uid == item.sop_instance_uid:
                return True
        return False

    def get_dcm_path(dcm_info):
        return dcm_info.dcm_path

    reader = sitk.ImageSeriesReader()
    if sort_by_distance:
        dcm_infos = []

        files = os.listdir(dcm_case_path)
        for file in files:
            file_path = osp.join(dcm_case_path, file)
            dcm = pydicom.dcmread(file_path, force=True)
            _series_instance_uid = dcm.SeriesInstanceUID
            _sop_instance_uid = dcm.SOPInstanceUID
            _instance_number = dcm.InstanceNumber
            _image_orientation_patient = dcm.ImageOrientationPatient
            _image_position_patient = dcm.ImagePositionPatient

            dcm_info = DcmInfo(
                file_path,
                _series_instance_uid,
                None,
                _sop_instance_uid,
                _instance_number,
                _image_orientation_patient,
                _image_position_patient,
            )

            if is_sop_instance_uid_exist(dcm_info, dcm_infos):
                continue

            dcm_infos.append(dcm_info)

        dcm_infos.sort(key=lambda x: x.slice_distance)
        dcm_series = list(map(get_dcm_path, dcm_infos))
    else:
        dcm_series = reader.GetGDCMSeriesFileNames(dcm_case_path)

    reader.SetFileNames(dcm_series)
    reader.SetNumberOfWorkUnits(16)
    sitk_image: sitk.Image = reader.Execute()

    if spacing is None:
        return sitk_image, None, None, None

    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    spacing = spacing[::-1]
    spacing_ratio = [original_spacing[i] / spacing[i] for i in range(3)]
    resampled_size = [int(original_size[i] * spacing_ratio[i]) - 1 for i in range(3)]

    resampled_mha = sitk.Resample(
        image1=sitk_image,
        size=resampled_size,  # type: ignore
        interpolator=sitk.sitkLinear,
        outputSpacing=spacing,
        outputPixelType=sitk.sitkInt16,
        outputOrigin=sitk_image.GetOrigin(),
        outputDirection=sitk_image.GetDirection(),
        transform=sitk.Transform(),
    )

    return resampled_mha, original_spacing, original_size, resampled_size


def LoadDcmAsSitkImage_JianYingOrder(dcm_case_path, spacing) -> tuple[
    sitk.Image,
    tuple[float, float, float] | None,
    tuple[int, int, int] | None,
    tuple[int, int, int] | None,
]:

    dcms = []
    dcm_paths = glob(osp.join(dcm_case_path, "*.dcm"))

    for dcm_path in dcm_paths:
        ds = pydicom.dcmread(dcm_path)
        if (0x20, 0x32) not in ds:  # (0020, 0032) Image Position (Patient)
            warnings.warn(
                Fore.YELLOW
                + f"ImagePosition Missing, Deprecating: {dcm_case_path}"
                + Style.RESET_ALL
            )
            return False
        dcms.append((dcm_path, ds[0x20, 0x32].value[-1]))

    else:
        dcms = sorted(dcms, key=lambda x: x[1], reverse=True)

    sorted_dcm_paths = [dcm[0] for dcm in dcms]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_dcm_paths)
    reader.SetNumberOfWorkUnits(8)
    sitk_image: sitk.Image = reader.Execute()

    if spacing is None:
        return sitk_image, None, None, None

    else:
        original_spacing = sitk_image.GetSpacing()
        original_size = sitk_image.GetSize()
        spacing = spacing[::-1]  # 对外接口尽量保持[D, H, W] 不要搞~~
        spacing_ratio = [original_spacing[i] / spacing[i] for i in range(3)]
        resampled_size = [
            int(original_size[i] * spacing_ratio[i]) - 1 for i in range(3)
        ]

        resampled_mha = sitk.Resample(
            image1=sitk_image,
            size=resampled_size,  # type: ignore
            interpolator=sitk.sitkLinear,
            outputSpacing=spacing,
            outputPixelType=sitk.sitkInt16,
            outputOrigin=sitk_image.GetOrigin(),
            outputDirection=sitk_image.GetDirection(),
            transform=sitk.Transform(),
        )

        return resampled_mha, original_spacing, original_size, resampled_size


def LoadMhaAnno(mha_root, patient, ori_spacing, out_spacing, resampled_size):
    anno = []
    for class_idx in range(0, 4):
        sitk_path = osp.join(mha_root, patient, patient + f"_{class_idx}.mha")
        if not osp.exists(sitk_path):
            return None, None

        sitk_image = sitk.ReadImage(sitk_path)
        sitk_image.SetSpacing(ori_spacing)

        resampled_mha = sitk.Resample(
            image1=sitk_image,
            size=resampled_size,
            interpolator=sitk.sitkNearestNeighbor,
            outputSpacing=out_spacing,
            outputPixelType=sitk.sitkUInt8,
            outputOrigin=sitk_image.GetOrigin(),
            outputDirection=sitk_image.GetDirection(),
            transform=sitk.Transform(),
        )
        array = sitk.GetArrayFromImage(resampled_mha)
        anno.append(array)

    anno_with_class_channel = np.stack(anno, axis=0)  # (Class, D, H, W)

    background_location = np.argwhere(anno_with_class_channel.sum(axis=0) == 0)
    background_seg_map = np.zeros_like(anno_with_class_channel[0])
    background_seg_map[
        background_location[:, 0], background_location[:, 1], background_location[:, 2]
    ] = 1
    anno_with_class_channel = np.concatenate(
        [background_seg_map[np.newaxis, ...], anno_with_class_channel], axis=0
    )

    anno_without_class_channel = np.argmax(anno_with_class_channel, axis=0)
    return anno_with_class_channel, anno_without_class_channel


def merge_masks(mhas: list[str]|list[sitk.Image], dtype=np.uint8) -> sitk.Image:
    """
    将所有类的掩码合并到一个掩码中并返回SimpleITK图像。

    :param mha_paths: 所有mha文件的路径列表
    :param class_index_map: 类名到索引的映射字典
    :return: 合并后的SimpleITK图像
    """
    # 初始化一个空的掩码图像
    merged_mask = None

    # 遍历mha文件路径列表中的每个文件
    for class_index, mha in enumerate(mhas):
        if isinstance(mha, str) and os.path.isfile(mha):
            mask = sitk.ReadImage(mha)
        elif isinstance(mha, sitk.Image):
            mask = mha
        else:
            raise NotImplementedError(f"Unsupported type: {type(mha)}")
        
        mask_array = sitk.GetArrayFromImage(mask)
        if merged_mask is None:
            merged_mask = np.zeros_like(mask_array)
        if np.any(merged_mask[mask_array == 1]):
            print(f"Warning: Overlapping masks detected for class {class_index + 1}.")
        merged_mask[mask_array == 1] = class_index + 1

    if merged_mask is None:
        raise ValueError("No mask found in the provided paths")

    # 将合并后的掩码转换为SimpleITK图像
    merged_mask_image = sitk.GetImageFromArray(merged_mask.astype(dtype))
    merged_mask_image.CopyInformation(mask)
    return merged_mask_image


def split_image_label_pairs_to_2d(image: sitk.Image, label: sitk.Image):
    """
    将image和label在最高维度上进行切分，迭代式返回slice-pair。

    :param image: SimpleITK图像
    :param label: SimpleITK图像
    :yield: (image_slice, label_slice) 对
    """
    # 一致性检查
    assert (
        image.GetSize() == label.GetSize()
    ), f"Image size {image.GetSize()} != Label size {label.GetSize()}"
    assert (
        image.GetSpacing() == label.GetSpacing()
    ), f"Image spacing {image.GetSpacing()} != Label spacing {label.GetSpacing()}"
    assert (
        image.GetOrigin() == label.GetOrigin()
    ), f"Image origin {image.GetOrigin()} != Label origin {label.GetOrigin()}"
    # assert image.GetDirection() == label.GetDirection(), f"Image direction {image.GetDirection()} != Label direction {label.GetDirection()}"

    # 将SimpleITK图像转换为NumPy数组
    image_array = sitk.GetArrayFromImage(image)
    label_array = sitk.GetArrayFromImage(label)

    # Z轴切片
    for i in range(len(image_array)):
        image_slice: np.ndarray = image_array[i]
        label_slice: np.ndarray = label_array[i]
        yield image_slice, label_slice
