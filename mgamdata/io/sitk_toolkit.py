import os
import os.path as osp
import pdb
import warnings
from collections.abc import Sequence
from glob import glob
from colorama import Style, Fore
from typing_extensions import deprecated

import pydicom
from pydicom import dicomio
import numpy as np
import SimpleITK as sitk


@deprecated("已在V2版本中实现性能改进")
def sitk_resample_to_spacing(image: sitk.Image, 
                             new_spacing:tuple[float, float, float], 
                             interpolator,
                             default_value=0.):
    """ 
        将一个sitk.Image对象重采样到指定的spacing。
        本方法的实现效率较差，且细节较多。
        在v2版本中，对部分功能转移由sitk内置方法实现，可靠性更高。
    """
    
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(),
                                           origin=image.GetOrigin() + offset, default_value=default_value)
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator,default_value=default_value)


def sitk_resample_to_spacing_v2(mha:sitk.Image, 
                                spacing:Sequence[float], 
                                field:str):
    """改进后的重采样方法。

    Args:
        mha (sitk.Image): 输入sitk.Image
        spacing (tuple[float,float,float]): 新的spacing
        field (str, optional): 重采样的对象。 可选'image', 'label', 'mask'.
                               本参数将决定插值方法和数据格式。
    
    Returns:
        sitk.Image: 重采样后的sitk.Image
    """
    
    assert field in ['image', 'label', 'mask'], "field must be one of ['image', 'label', 'mask']"
    
    # 计算重采样后的Spacing
    spacing = spacing[::-1]
    original_size = mha.GetSize()
    original_spacing = mha.GetSpacing()
    spacing_ratio = [original_spacing[i]/spacing[i] for i in range(3)]
    resampled_size = [int(original_size[i] * spacing_ratio[i])-1 for i in range(3)]
    
    # 执行重采样
    mha_resampled = sitk.Resample(
        image1=mha,
        size=resampled_size,# type:ignore
        interpolator=sitk.sitkBSpline5 if field == 'image' else sitk.sitkNearestNeighbor,
        outputSpacing=spacing,
        outputPixelType=sitk.sitkInt16 if field == 'image' else sitk.sitkUInt8,
        outputOrigin=mha.GetOrigin(),
        outputDirection=mha.GetDirection(),
        transform=sitk.Transform(),
    )
    
    return mha_resampled


def sitk_resample_to_image(image:sitk.Image, 
                           reference_image:sitk.Image, 
                           field='image',
                           default_value=0.):
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
    
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(sitk.sitkLinear if field == 'image' else sitk.sitkNearestNeighbor)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetOutputPixelType(sitk.sitkInt16 if field == 'image' else sitk.sitkUInt8)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)


def sitk_resample_to_size(image, new_size:list[float], field='image'):
    new_size = new_size[::-1]
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    if len(new_size) == 2:
        new_size = new_size + [original_size[-1]]
    new_spacing = np.divide(original_spacing, np.divide(new_size, original_size))
    resampled = sitk.Resample(
        image1=image,
        size=new_size,
        interpolator=sitk.sitkLinear if field == 'image' else sitk.sitkNearestNeighbor,
        outputSpacing=new_spacing,
        outputPixelType=sitk.sitkInt16 if field == 'image' else sitk.sitkUInt8,
        outputOrigin=image.GetOrigin(),
        outputDirection=image.GetDirection(),
        transform=sitk.Transform(),
    )
    return resampled


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    pdb.set_trace()
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float32).T * default_value, isVector=False)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def calculate_origin_offset(new_spacing, old_spacing):
    return np.subtract(new_spacing, old_spacing) / 2.0


def LoadDcmAsSitkImage_EngineeringOrder(dcm_case_path, spacing, sort_by_distance=True
    ) -> tuple[sitk.Image, 
         tuple[float, float, float]|None, 
         tuple[int, int, int]|None,
         tuple[int, int, int]|None
    ]:
    # Spacing: [D, H, W]
    
    class DcmInfo(object):
        def __init__(self, dcm_path, series_instance_uid, 
                     acquisition_number, sop_instance_uid, instance_number,
                     image_orientation_patient, image_position_patient):
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
            normal = [self.image_orientation_patient[1] * self.image_orientation_patient[5] -
                      self.image_orientation_patient[2] * self.image_orientation_patient[4],
                      self.image_orientation_patient[2] * self.image_orientation_patient[3] -
                      self.image_orientation_patient[0] * self.image_orientation_patient[5],
                      self.image_orientation_patient[0] * self.image_orientation_patient[4] -
                      self.image_orientation_patient[1] * self.image_orientation_patient[3]]

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

            dcm_info = DcmInfo(file_path, _series_instance_uid, None, _sop_instance_uid,
                               _instance_number, _image_orientation_patient, _image_position_patient)

            if is_sop_instance_uid_exist(dcm_info, dcm_infos):
                continue

            dcm_infos.append(dcm_info)

        dcm_infos.sort(key=lambda x: x.slice_distance)
        dcm_series = list(map(get_dcm_path, dcm_infos))
    else:
        dcm_series = reader.GetGDCMSeriesFileNames(dcm_case_path)

    reader.SetFileNames(dcm_series)
    reader.SetNumberOfWorkUnits(16)
    sitk_image:sitk.Image = reader.Execute()
    
    if spacing is None:
        return sitk_image, None, None, None
    
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    spacing = spacing[::-1]
    spacing_ratio = [original_spacing[i]/spacing[i] for i in range(3)]
    resampled_size = [int(original_size[i] * spacing_ratio[i])-1 for i in range(3)]
    
    resampled_mha = sitk.Resample(
            image1=sitk_image,
            size=resampled_size,# type: ignore
            interpolator=sitk.sitkLinear,
            outputSpacing=spacing,
            outputPixelType=sitk.sitkInt16,
            outputOrigin=sitk_image.GetOrigin(),
            outputDirection=sitk_image.GetDirection(),
            transform=sitk.Transform(),
        )
    
    return resampled_mha, original_spacing, original_size, resampled_size


def LoadDcmAsSitkImage_JianYingOrder(dcm_case_path, spacing
    ) -> tuple[sitk.Image, 
               tuple[float, float, float]|None, 
               tuple[int, int, int]|None,
               tuple[int, int, int]|None
    ]:
    
    dcms = []
    dcm_paths = glob(osp.join(dcm_case_path, '*.dcm'))

    for dcm_path in dcm_paths:
        ds = pydicom.dcmread(dcm_path)
        if (0x20, 0x32) not in ds: # (0020, 0032) Image Position (Patient)
            warnings.warn(Fore.YELLOW + 
                        f"ImagePosition Missing, Deprecating: {dcm_case_path}" +
                        Style.RESET_ALL)
            return False
        dcms.append((dcm_path, ds[0x20, 0x32].value[-1]))
    
    else:
        dcms = sorted(dcms, key=lambda x: x[1], reverse=True)
    
    sorted_dcm_paths = [dcm[0] for dcm in dcms]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_dcm_paths)
    reader.SetNumberOfWorkUnits(8)
    sitk_image:sitk.Image = reader.Execute()

    if spacing is None:
        return sitk_image, None, None, None
    
    else:
        original_spacing = sitk_image.GetSpacing()
        original_size = sitk_image.GetSize()
        spacing = spacing[::-1] # 对外接口尽量保持[D, H, W] 不要搞~~
        spacing_ratio = [original_spacing[i]/spacing[i] for i in range(3)]
        resampled_size = [int(original_size[i] * spacing_ratio[i])-1 for i in range(3)]
        
        resampled_mha = sitk.Resample(
                image1=sitk_image,
                size=resampled_size,# type: ignore
                interpolator=sitk.sitkLinear,
                outputSpacing=spacing,
                outputPixelType=sitk.sitkInt16,
                outputOrigin=sitk_image.GetOrigin(),
                outputDirection=sitk_image.GetDirection(),
                transform=sitk.Transform(),
            )
        
        return resampled_mha, original_spacing, original_size, resampled_size


def LoadDcmAsSitkImage(mode:str, dcm_case_path:str, spacing:tuple[float, float, float]):
    assert mode.lower() in ['engineering', 'jianying'], "mode must be one of ['engineering', 'jianying']"

    if mode.lower() == 'engineering':
        return LoadDcmAsSitkImage_EngineeringOrder(dcm_case_path, spacing)
    else:
        return LoadDcmAsSitkImage_JianYingOrder(dcm_case_path, spacing)


def LoadMhaAnno(mha_root, patient, ori_spacing, out_spacing, resampled_size):
    anno = []
    for class_idx in range(0,4):
        sitk_path = osp.join(mha_root, patient, patient+f"_{class_idx}.mha")
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
    
    anno_with_class_channel = np.stack(anno, axis=0)    # (Class, D, H, W)
    
    background_location = np.argwhere(anno_with_class_channel.sum(axis=0) == 0)
    background_seg_map = np.zeros_like(anno_with_class_channel[0])
    background_seg_map[background_location[:, 0], background_location[:, 1], background_location[:, 2]] = 1
    anno_with_class_channel = np.concatenate([background_seg_map[np.newaxis, ...], anno_with_class_channel], axis=0)
    
    anno_without_class_channel = np.argmax(anno_with_class_channel, axis=0)
    return anno_with_class_channel, anno_without_class_channel


def merge_masks(mha_paths: list[str], 
                class_index_map: dict[str, int],
                dtype=np.uint8
                ) -> sitk.Image:
    """
    将所有类的掩码合并到一个掩码中并返回SimpleITK图像。
    
    :param mha_paths: 所有mha文件的路径列表
    :param class_index_map: 类名到索引的映射字典
    :return: 合并后的SimpleITK图像
    """
    # 初始化一个空的掩码图像
    merged_mask = None
    
    # 遍历mha文件路径列表中的每个文件
    for seg_file_path in mha_paths:
        if os.path.isfile(seg_file_path):
            class_name = os.path.basename(seg_file_path)[:-4]
            class_index = class_index_map.get(class_name)
            if class_index is None:
                raise ValueError(f"Class name {class_name} not found in class_index_map: {seg_file_path}")
            
            # 读取掩码文件
            mask = sitk.ReadImage(seg_file_path)
            mask_array = sitk.GetArrayFromImage(mask)
            # 初始化合并掩码
            if merged_mask is None:
                merged_mask = np.zeros_like(mask_array)
            # 将当前类的掩码添加到合并掩码中
            merged_mask[mask_array == 1] = class_index
    
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
    assert image.GetSize() == label.GetSize(), f"Image size {image.GetSize()} != Label size {label.GetSize()}"
    assert image.GetSpacing() == label.GetSpacing(), f"Image spacing {image.GetSpacing()} != Label spacing {label.GetSpacing()}"
    assert image.GetOrigin() == label.GetOrigin(), f"Image origin {image.GetOrigin()} != Label origin {label.GetOrigin()}"
    # assert image.GetDirection() == label.GetDirection(), f"Image direction {image.GetDirection()} != Label direction {label.GetDirection()}"
    
    # 将SimpleITK图像转换为NumPy数组
    image_array = sitk.GetArrayFromImage(image)
    label_array = sitk.GetArrayFromImage(label)
    
    # Z轴切片
    for i in range(len(image_array)):
        image_slice:np.ndarray = image_array[i]
        label_slice:np.ndarray = label_array[i]
        yield image_slice, label_slice

