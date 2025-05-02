import SimpleITK as sitk
import numpy as np

def remap_labels(sitk_image:sitk.Image, remap_dict:dict[int, int]):
    """
    根据映射表 remap_dict 对 sitk_image 的标签进行重映射。

    参数:
        sitk_image (sitk.Image): 输入的标签图像（label map）。
        remap_dict (dict): 标签映射表，key 为原始标签，value 为新标签。

    返回:
        sitk.Image: 标签已重映射的新 SimpleITK 图像。
    """
    arr = sitk.GetArrayFromImage(sitk_image)
    remapped_arr = np.copy(arr)
    for old_label, new_label in remap_dict.items():
        remapped_arr[arr == old_label] = new_label
    remapped_image = sitk.GetImageFromArray(remapped_arr)
    remapped_image.CopyInformation(sitk_image)
    return remapped_image
