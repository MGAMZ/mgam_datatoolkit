import math
import logging
import os
import pdb
import os.path as osp
from typing import Dict
from typing import List
from typing import Tuple
from typing import Sequence


import torch
import pydicom
import nrrd
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_hist
from skimage.transform import warp

from mmseg.registry import DATASETS
from mmseg.registry import TRANSFORMS
from mmseg.registry import HOOKS
from mmseg.datasets.basesegdataset import BaseSegDataset

from mmseg.structures import SegDataSample
from mmseg.engine.hooks import SegVisualizationHook
from mmengine.runner import Runner

from mmengine.utils import ManagerMixin
from mmengine.logging import print_log
from mmengine import ConfigDict
from mmcv.transforms.base import BaseTransform

import DistortionAugment
from .CQKGastricCancerCTBackend import CQKGastricCancerCT
from .lmdb_GastricCancer import LMDB_DataBackend




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


@DATASETS.register_module()
class GastricCancer_2023(BaseSegDataset):
	METAINFO = dict(
		classes=('normal','cancer'),
		palette=[[0], [255]],
	)

	def __init__(self, database_args:dict, debug:bool=False, *args, **kwargs):
		if database_args.get("lmdb_backend_proxy", None):
			assert isinstance(database_args["lmdb_backend_proxy"], ConfigDict), "[DATASET] lmdb_backend_proxy must be a dict"
			lmdb_backend_proxy:LMDB_MP_Proxy = TRANSFORMS.build(database_args["lmdb_backend_proxy"])
			lmdb_service:LMDB_DataBackend = lmdb_backend_proxy()
			meta_key_name = "REGISTRY_"+os.path.basename(database_args["metadata_ckpt"])
			meta_dict = lmdb_service.get(meta_key_name)
			if meta_dict:
				database_args["metadata_ckpt"] = meta_dict
			else:
				FileNotFoundError(f"metadata_ckpt {meta_key_name} not found from lmdb_backend_proxy")

		self.debug = debug
		self._database_args = database_args
		self._DATABASE = CQKGastricCancerCT(**database_args)

		super().__init__(*args, **kwargs)
	
	def load_data_list(self):
		print_log(f"[Dataset] 索引数据集 | split:{self._database_args['split']} | p:{self._database_args['num_positive_img']} | n:{self._database_args['num_negative_img']} | d:{self._database_args['minimum_negative_distance']}", 
				  "current", logging.INFO)
		data_list = self._DATABASE.MMSEG_Segmentation_PosNegEnhance(
						self._database_args['split'],
						self._database_args['num_positive_img'],
						self._database_args['num_negative_img'],
						self._database_args['minimum_negative_distance']
					)	# list[dict]
		print_log(f"[Dataset] 数据集索引完成 | split:{self._database_args['split']} | num_sam:{len(data_list)}", 
				  "current", logging.INFO)
		return data_list[:32] if self.debug else data_list



class CTSegDatasetWithDistortionCache(GastricCancer_2023):
	def __init__(self, distortion_args:dict, *args, **kwgs):
		super().__init__(*args, **kwgs)
		self.distorter = DistortionAugment.Distortion(**distortion_args)

	def _pregenerate_distortion(self, data_list:List[str]):
		return self.distorter.multiprocess_distort(data_list)

	def load_data_list(self):
		data_list = super().load_data_list()
		self.data_cache = self._pregenerate_distortion(data_list)
		return self.data_cache




@TRANSFORMS.register_module()
class RadialStretch(BaseTransform):
	def __init__(self, *args, **kwgs) -> None:
		self.exec = DistortionAugment.RadialStretch(*args, **kwgs)
	
	def transform(self, results: dict) -> dict:
		results['img'] = self.exec.stretch(results['img'])
		if 'gt_seg_map' in results and self.exec.mmseg_stretch_seg_map:
			results['gt_seg_map'] = self.exec.stretch(results['gt_seg_map'])
		return results


@TRANSFORMS.register_module()
class Distortion(BaseTransform):
	def __init__(self, *args, **kwgs) -> None:
		self.exec = DistortionAugment.Distortion(*args, **kwgs)
	
	def transform(self, results: dict) -> dict:
		# 在开始时或每隔一段时间，刷新映射矩阵
		if not self.exec.const:
			if self.exec.refresh_counter % self.exec.refresh_interval == 0:
				self.exec.refresh_affine_map()
		
		results['img'] = warp(image=results['img'], 
							  inverse_map=self.exec.tform, 
							  order=1,
							  preserve_range=True)
		if 'gt_seg_map' in results:
			results['gt_seg_map'] = warp(image=results['gt_seg_map'], 
								inverse_map=self.exec.tform, 
								order=0,
								preserve_range=True)
		
		self.exec.refresh_counter += 1
		return results












if __name__ == '__main__':
	dcm = pydicom.dcmread(r".\img\柏家荣\ImageFileName000.dcm")

	
	lmdb_args = {'dataset_root': "../2023_Med_CQK", 
				'lmdb_path': "../2023_Med_CQK/lmdb_database", 
				'mode': "normal"}
	backend = LMDB_MP_Proxy.get_instance('test', lmdb_args=lmdb_args)
	backend = LMDB_DataBackend(**lmdb_args)
	backend = LMDB_MP_Proxy('test', lmdb_args)
	backend.database_test()






