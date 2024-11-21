import os.path as osp
import pdb
from abc import abstractmethod
from prettytable import PrettyTable
from collections import OrderedDict

import cv2
import torch
import numpy as np
from skimage.exposure import equalize_hist
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import mmcv
import mmengine
from mmengine.dist.utils import master_only
from mmengine.logging import print_log, MMLogger
from mmengine.runner import Runner
import mmengine.structures
from mmseg.evaluation.metrics import IoUMetric
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.visualization import SegLocalVisualizer
from mmseg.structures import SegDataSample
from mmcv.transforms import BaseTransform



class HistogramEqualization(BaseTransform):
    def __init__(self, image_size:tuple, ratio:float):
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
    
    
    def transform(self, results:dict) -> dict:
        assert isinstance(results['img'], list) 
        for i, image in enumerate(results['img']):
            results['img'][i] = self.RoI_HistEqual(image)
        return results


class IoUMetric_PerClass(IoUMetric):
    def compute_metrics(self, results: list) -> dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict[str, float]: The computed metrics. The keys are the names of
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
            ret_metric: [format(v, '.2f') for v in ret_metric_value * 100]
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


class SegVisualizationHook_Base(SegVisualizationHook):
    @abstractmethod
    def _get_source_image(self, data_sample: SegDataSample) -> np.ndarray:
        ...
    
    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: list[SegDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx
        
        # Visualize only the first data
        window_name = f'val_{osp.basename(outputs[0].img_path)}'
        img = self._get_source_image(outputs[0])
        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: list[SegDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        for data_sample in outputs:
            self._test_index += 1
            window_name = f'test_{osp.basename(data_sample.img_path)}'

            img = self._get_source_image(data_sample)
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index)


class SegViser(SegLocalVisualizer):
    def __init__(self, name, draw_heatmap:bool=False, draw_others:bool=True,*args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.draw_heatmap = draw_heatmap
        self.draw_others = draw_others
        if draw_heatmap:
            self.fig = plt.subplots(1, 2, figsize=(10, 5))

    def _draw_heatmap(self,
                      image: np.ndarray,
                      gt_seg: mmengine.structures.PixelData,
                      seg_logit: mmengine.structures.PixelData) -> np.ndarray:
        gt_seg_array = gt_seg.data.squeeze().cpu().numpy()
        seg_logit_array = seg_logit.data.squeeze().cpu().numpy()
        assert gt_seg_array.shape == seg_logit_array.shape, f"Shape mismatch: gt_seg_array {gt_seg_array.shape} != sem_seg_array {sem_seg_array.shape}"
        assert image.shape[:2] == gt_seg_array.shape[:2], f"Shape mismatch: image {image.shape[:2]} != gt_seg_array {gt_seg_array.shape[:2]}"
        
        gt_normalized = gt_seg_array / gt_seg_array.max()
        gt_rgb = plt.get_cmap('rainbow')(gt_normalized)[:, :, :3] * 255
        gt_heatmap = cv2.addWeighted(image, self.alpha, gt_rgb.astype(np.uint8), 0.5, 0)
        
        seg_logit_normalized = Normalize(vmin=0, vmax=1)(seg_logit_array)
        seg_logit_rgb = plt.get_cmap('rainbow')(seg_logit_normalized)[:, :, :3] * 255
        seg_logit_heatmap = cv2.addWeighted(image, self.alpha, seg_logit_rgb.astype(np.uint8), 0.5, 0)
        
        # concat two heatmap
        heatmap = np.concatenate([gt_heatmap, seg_logit_heatmap], axis=1)
        return heatmap

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: SegDataSample|None = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: str|None = None,
            step: int = 0,
            with_labels: bool|None = True) -> None:
        
        if self.draw_heatmap:
            heatmap = self._draw_heatmap(image, data_sample.gt_sem_seg, data_sample.seg_logits)
            self.add_image("heatmap_" + name, heatmap, step)
        if self.draw_others:
            super().add_datasample(name, image, data_sample, draw_gt, draw_pred, show, wait_time, out_file, step, with_labels)


class SegVisHook_Vanilla(SegVisualizationHook_Base):
    def _get_source_image(self, data_sample: SegDataSample) -> np.ndarray:
        img_path = data_sample.img_path
        img_bytes = mmengine.fileio.get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        return img


class SegVisHook_Npz(SegVisualizationHook_Base):
    def _get_source_image(self, data_sample: SegDataSample) -> np.ndarray:
        img = np.load(data_sample.img_path)['img']
        return img
