import os
import pdb
import math
import logging
from typing_extensions import Literal

import cv2
import torch
import numpy as np
from scipy.ndimage import map_coordinates
from torch import Tensor
from mmcv.transforms import BaseTransform
from mmengine.logging import print_log
from mmseg.datasets.transforms import PackSegInputs as _PackSegInputs



def rectangular_to_polar(x, y, center_x, center_y):
    """
    直角坐标由0开始计数
    标准直角坐标系输入: x,y
    极点的直角坐标: center_x, center_y
    
    radius: 极径
    angle: 极角 弧度制
    """
    # 使用numpy计算半径
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # 使用scipy计算角度
    angle = np.arctan2(y - center_y, x - center_x)
    
    return radius, angle


def polar_to_rectangular(radius, angle, center_x, center_y):
    """
    直角坐标由0开始计数
    radius: 极径
    angle: 极角 弧度制
    center_x, center_y: 极点的直角坐标

    x,y: 直角坐标
    """
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    
    return x, y


class RadialStretch:
    def __init__(self,
                 CornerFactor=1, 
                 GlobalFactor=1, 
                 in_array_shape:tuple=None,
                 direction:str="out", 
                 mmseg_stretch_seg_map:bool=True,
                 stretch_num_workers:int=8):
        assert CornerFactor>=0 and GlobalFactor>=1, "[Dataset] Projection Map Init Error: CornerFactor must >=0, GlobalFactor must >=1"
        assert in_array_shape[0]==in_array_shape[1], "[Dataset] Projection Map Init Error: input image must be square"
        assert direction in ['out', 'in'], "[Dataset] Stretch Direction can only be out or in. Out mean Stretch out to a square. In is its reverse operation"
        
        self.CornerFactor = CornerFactor            # 控制边角拉伸的强度
        self.GlobalFactor = GlobalFactor            # 控制全局放大的强度
        self.in_array_shape = in_array_shape        # 输入矩阵的尺寸
        self.direction = direction                  # 控制拉伸或反拉伸, out为拉伸向外
        self.mmseg_stretch_seg_map = mmseg_stretch_seg_map      # 是否拉伸标签图
        self.stretch_num_workers = stretch_num_workers # 自带的多进程拉伸时的进程数
        self._cache_map()

    def _cache_map(self):
        print_log("[Dataset] 正在缓冲拉伸映射矩阵", "current", logging.INFO)
        # 输出矩阵对应每个点都有一个映射坐标
        map_height, map_width = self.in_array_shape
        self.proj_map = np.zeros(shape=(map_height, map_width, 2), dtype=np.uint16)
        # 遍历每个像素, 生成映射矩阵
        for y in range(map_height):
            for x in range(map_width):
                self.proj_map[y, x] = self.CoordinateMapping(y, x)
        self.proj_map.setflags(write=False) # 锁定映射矩阵
        print_log("[Dataset] 已缓冲拉伸映射矩阵", "current", logging.INFO)

    # 计算该极角的拉伸倍数, 仅适用于正方形输入输出。
    # 输入的是当前映射矩阵的极角, 弧度制。
    # 输出该极角下映射矩阵的极径拉伸倍数，应当在外部与当前映射矩阵的极径相乘，得到source的极径。
    def stretch_factor(self, map_radians):
        # 输入为弧度制
        map_radians = abs(map_radians) % (math.pi/2) # 以90为周期，关于Y轴对称

        # Deprecated: 线性角度映射，有突变点
        # if angle > 45:
        #     angle = 90 - angle  # 周期内中心对称

        # 渐进Cos角度映射，周期内中心对称，但对称模式改变。
        # 设立直角坐标系，X轴及X轴上方有效，X轴代表源角度，Y轴代表映射角度（输出至形变参量的计算）
        # 
        angle = (math.pi/8) * (1 - math.cos(4*map_radians))

        # 形变参量
        radial_factor = 1 / (math.cos(angle)**self.CornerFactor)
        if self.direction == 'out':     # 图像拉伸方向为向外
            radial_factor = 1 / radial_factor
            global_factor = 1 / self.GlobalFactor
        elif self.direction == 'in':    # 图像拉伸方向为向内
            radial_factor = radial_factor
            global_factor = self.GlobalFactor
        # 最终缩放参数 = 该方向上的形变 * 整体缩放
        factor = radial_factor * global_factor
        
        return factor

    # 输入处理后矩阵索引，返回源矩阵索引
    def CoordinateMapping(self, map_Y, map_X):
        # XY均由0开始计数
        # 数组存储图片时，原点位于左上角，这里将Y轴坐标反置，使原点移动至左下角
        true_map_y = self.in_array_shape[0] - map_Y
        # 映射矩阵极点默认为映射矩阵中心
        map_center_y, map_center_x = self.in_array_shape[0]/2, self.in_array_shape[1]/2
        # 输入图像极点默认为输入图像中心
        source_center_y, source_center_x = self.in_array_shape[0]/2, self.in_array_shape[1]/2
        # 获取该点在映射矩阵中的极坐标
        radius, angle = rectangular_to_polar(map_X, true_map_y, map_center_y, map_center_x)
        # 计算该极角的拉伸倍数
        stretch_factor_of_this_angle = self.stretch_factor(angle)
        # 转换回直角坐标, 寻找源坐标
        source_x, true_source_y = polar_to_rectangular(radius*stretch_factor_of_this_angle, angle, source_center_x, source_center_y)
        # 四舍五入, 坐标限制
        source_x, true_source_y = np.clip(round(source_x), 1, self.in_array_shape[1]), np.clip(round(true_source_y), 1, self.in_array_shape[0])
        # Y轴坐标反置的恢复，顺带恢复到索引值域
        source_y = self.in_array_shape[0] - true_source_y
        # X轴恢复到索引值域
        source_x -= 1
        return source_y, source_x

    # 单输入执行拉伸
    def stretch(self, image_matrix, type=Literal['img', 'label'], pad_val=None, seg_pad_val=None):
        # 当参数等效为无拉伸时，直接返回
        if self.CornerFactor==0 and self.GlobalFactor==1:
            return image_matrix
        
        out_shape = self.proj_map.shape[:-1]
        # numpy映射比tensor快一个数量级以上
        # 创建与输入矩阵相同大小的零矩阵
        if isinstance(image_matrix, Tensor):
            if image_matrix.dtype == torch.uint8:
                stretched_matrix = np.zeros(out_shape, dtype=np.uint8)
            else:
                stretched_matrix = np.zeros(out_shape, dtype=np.float32)
        elif isinstance(image_matrix, np.ndarray):
            stretched_matrix = np.zeros(out_shape, dtype=image_matrix.dtype)
        else:
            raise RuntimeError(f"Stretch get unsupported type: {type(stretched_matrix)}")

        # 映射
        try:
            map_coordinates(image_matrix, 
                            self.proj_map.transpose(2,0,1), 
                            output=stretched_matrix, 
                            mode='constant', 
                            cval=pad_val if type=='img' else seg_pad_val, 
                            prefilter=True)
        except Exception as e:
            pdb.set_trace()

        if isinstance(image_matrix, Tensor):
            stretched_matrix = torch.from_numpy(stretched_matrix).to(dtype=image_matrix.dtype,
                                                                     non_blocking=True)
        return stretched_matrix

    def calculate_density_factor_map(self):
        """
        计算每个像素位置的信息密集度因子分布图
        信息密集度因子为对应方向拉伸倍数的倒数
        """
        print_log("[Dataset] 正在计算信息密集度因子分布图", "current", logging.INFO)
        map_height, map_width = self.in_array_shape
        density_map = np.zeros(shape=(map_height, map_width), dtype=np.float32)
        # 映射矩阵极点默认为映射矩阵中心
        map_center_y, map_center_x = self.in_array_shape[0]/2, self.in_array_shape[1]/2
        # 遍历每个像素，计算其信息密集度因子
        for y in range(map_height):
            for x in range(map_width):
                # 获取真实的Y坐标（坐标系转换）
                true_y = self.in_array_shape[0] - y
                # 获取该点在映射矩阵中的极坐标
                radius, angle = rectangular_to_polar(x, true_y, map_center_x, map_center_y)
                # 计算该极角的拉伸倍数
                stretch_factor_value = self.stretch_factor(angle)
                # 信息密集度因子为拉伸倍数的倒数
                density_factor = 1.0 / stretch_factor_value
                # 存储到密度图中
                density_map[y, x] = density_factor
        
        print_log("[Dataset] 信息密集度因子分布图计算完成", "current", logging.INFO)
        return density_map

    def get_density_factor_map(self):
        """
        获取信息密集度因子分布图，如果不存在则计算
        返回与输入图像同样大小的numpy数组
        """
        if not hasattr(self, '_density_map'):
            self._density_map = self.calculate_density_factor_map()
        return self._density_map


class LoadDensityMap(BaseTransform):
    def transform(self, results:dict):
        density_map_path = results['img_path'].replace('image', "density")
        if os.path.exists(density_map_path):
            results["density"] = cv2.imread(density_map_path, cv2.IMREAD_UNCHANGED)
            results["seg_fields"].append("density")
        return results


class PackSegInputs(_PackSegInputs):
    def transform(self, results:dict):
        packed_results = super().transform(results)
        
        if "density" in results.keys():
            density:np.ndarray = results["density"]
            if density.ndim == 2:
                density = np.expand_dims(density, -1)
            density = torch.from_numpy(density.transpose(2, 0, 1))
            packed_results['data_samples'].set_field(density, "density")
            
        return packed_results

