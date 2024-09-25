import os.path as osp
from prettytable import PrettyTable
from typing import Tuple, List, Dict, OrderedDict

import cv2
import torch
import numpy as np
from skimage.exposure import equalize_hist

from mmengine.logging import print_log, MMLogger
from mmseg.evaluation.metrics import IoUMetric
from mmcv.transforms import BaseTransform




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



class IoUMetric_PerClass(IoUMetric):
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect:torch.Tensor = sum(results[0])
        total_area_union:torch.Tensor = sum(results[1])
        total_area_pred_label:torch.Tensor = sum(results[2])
        total_area_label:torch.Tensor = sum(results[3])
        
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)
        class_names = self.dataset_meta['classes']

        # class averaged table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        class_metrics = OrderedDict({
            ret_metric: format(ret_metric_value, '.3f')
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        class_metrics.update({'Class': class_names})
        class_metrics.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in class_metrics.items():
            class_table_data.add_column(key, val)
        
        # provide per class results for logger hook
        metrics['PerClass'] = class_metrics

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

