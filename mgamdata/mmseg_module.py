from typing import Tuple, List, Dict

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from skimage.exposure import equalize_hist



class HistogramEqualization(BaseTransform):
    def __init__(self, image_size:Tuple, ratio:float):
        assert image_size[0] == image_size[1], 'Only support square shape for now.'
        assert ratio<1, 'RoI out of bounds'
        self.RoI = self.create_circle_in_square(image_size[0], image_size[0]*ratio)
        self.nbins = image_size[0]
    
    @staticmethod
    def create_circle_in_square(size:int, radius:int) -> np.ndarray:
        # 创建一个全0的正方形ndarray
        square = np.zeros((size, size))
        # 计算中心点的坐标
        center = size // 2
        # 计算每个元素到中心的距离
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        # 如果距离小于或等于半径，将该元素设置为1
        square[mask] = 1
        return square
    
    
    def RoI_HistEqual(self, image:np.ndarray):
        dtype_range = np.iinfo(image)
        normed_image = equalize_hist(image, nbins=self.nbins, mask=self.RoI)
        normed_image = (normed_image*dtype_range.max).astype(image.dtype)
        return normed_image
    
    
    def transform(self, results:Dict) -> Dict:
        assert isinstance(results['img'], List) 
        for i, image in enumerate(results['img']):
            results['img'][i] = self.RoI_HistEqual(image)
        return results






