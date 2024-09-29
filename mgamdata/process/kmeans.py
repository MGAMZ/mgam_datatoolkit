import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans

from ..dataset.RenJi_Sarcopenia.meta import SARCOPENIA_FOREGROUND_CLASSES
from .GeneralPreProcess import WindowSet



def binary_kmeans_segment(array:np.ndarray, mask=None):
    """ Kmeans二分类分割。
        编写这个函数最开始的目的是用于从肌肉中分出肌间脂肪来的。
    
    Args:
        array: np.ndarray
            The array to segment.
        mask: np.ndarray (Optional)
            A binary mask of the same shape as input array.
            Only the pixels where mask is True will be used for clustering.
    
    Returns:
        binary_threshold: float
            The threshold value that can be used to segment the input array into two classes.
    """
    
    if mask is not None:
        array = array[mask].reshape(-1, 1)
    kmeans_result = KMeans(n_clusters=2, random_state=0).fit(array)
    higher_value_class_idx = np.argmax(kmeans_result.cluster_centers_, axis=0)
    binary_threshold = array[kmeans_result.labels_ == higher_value_class_idx].min()
    return binary_threshold



def sarcopenia_muscle_subsegmentation(array:np.ndarray, basic_seg_mask:np.ndarray):
    post_seg_mask = basic_seg_mask.copy()
    array = WindowSet(40, 400).transform({'img': array})['img']
    
    for class_idx in [1, 2]:
        parent_class_mask = post_seg_mask==class_idx
        if parent_class_mask.any():
            threshold = binary_kmeans_segment(array, parent_class_mask)
            subclass_fat_mask = (array < threshold) * parent_class_mask
            # 将找到的肌间脂肪指定为新的类，类序号为原类序号+4，详见类定义。
            post_seg_mask[subclass_fat_mask] = class_idx + len(SARCOPENIA_FOREGROUND_CLASSES)

    return post_seg_mask


def sarcopenia_muscle_subsegmentation_fromITK(ITK_image_path:str, ITK_mask_path:str):
    """ 从ITK文件中读取数据并进行肌间脂肪分割
    
    Args:
        ITK_image_path: str
            The path to the ITK image file.
        ITK_mask_path: str
            The path to the ITK mask file.
    
    Returns:
        post_seg_mask: np.ndarray
            The segmented mask.
    """
    try:
        image = sitk.ReadImage(ITK_image_path)
    except Exception as e:
        raise RuntimeError(f"Read image File({ITK_image_path}) Failed: {e}")
    try:
        mask = sitk.ReadImage(ITK_mask_path)
    except Exception as e:
        raise RuntimeError(f"Read label File({ITK_mask_path}) Failed: {e}")
    
    image_data = sitk.GetArrayFromImage(image)
    mask_data = sitk.GetArrayFromImage(mask)
    
    post_seg_mask = sarcopenia_muscle_subsegmentation(image_data, mask_data)
    
    post_seg_itk_mask = sitk.GetImageFromArray(post_seg_mask)
    post_seg_itk_mask.CopyInformation(mask)
    return post_seg_itk_mask
