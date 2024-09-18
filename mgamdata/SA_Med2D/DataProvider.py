import os
import pdb
from typing import Dict, List, OrderedDict, Tuple, Type
from colorama import Fore, Style

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.logging import MMLogger, print_log
# mmseg可视化Hook设定
from mmseg.datasets.transforms import LoadBiomedicalImageFromFile
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.evaluation.metrics import IoUMetric
from prettytable import PrettyTable

from .DatasetBackend import DatasetBackend_GlobalProxy




class SegVisualizationHook_SA_Med2D(SegVisualizationHook):
    def __init__(self, reshape_size:Tuple[int,int] , *args, **kwargs):
        self.reshape_size = reshape_size
        super().__init__(*args, **kwargs)
    
    def _after_iter(self,
                    runner,
                    batch_idx,
                    data_batch,
                    outputs,
                    mode) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train' or mode == 'val':
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = os.path.join(
                            output.img_path[0], # type: ignore
                            f"image_{output.img_path[1]}.png") # type: ignore
                img = Load_SA_Med2D.load_png(img_path)
                img = cv2.resize(img, self.reshape_size, interpolation=cv2.INTER_NEAREST)
                window_name = f'{mode}_{os.path.basename(img_path)}'

                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)



class IoU_SupportsMultipleIgnoreIndex(IoUMetric):
    def __init__(self, valid_label_on_summarizer=None, **kwargs):
        if valid_label_on_summarizer is None:
            self.valid_label_on_summarizer = None
        elif isinstance(valid_label_on_summarizer, Tuple):
            self.valid_label_on_summarizer = torch.tensor(valid_label_on_summarizer)
        else:
            self.valid_label_on_summarizer = 'dynamic'
        super().__init__(**kwargs)
    
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
            logger.info(f'results are saved to {os.path.dirname(self.output_dir)}')
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
        
        if self.valid_label_on_summarizer is not None:
            # 去除没有gt的label
            if self.valid_label_on_summarizer == 'dynamic':
                NonZero_Class_Index = torch.where(total_area_label != 0)[0]
            # 或者通过外部参数指定需要去除的label index
            else:
                NonZero_Class_Index = self.valid_label_on_summarizer
            total_area_intersect = torch.gather(total_area_intersect, 0, NonZero_Class_Index)
            total_area_union = torch.gather(total_area_union, 0, NonZero_Class_Index)
            total_area_pred_label = torch.gather(total_area_pred_label, 0, NonZero_Class_Index)
            total_area_label = torch.gather(total_area_label, 0, NonZero_Class_Index)
        
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        if self.valid_label_on_summarizer is not None:
            # 对应地去除class名称
            class_names = [self.dataset_meta['classes'][i] for i in NonZero_Class_Index]
        else:
            class_names = self.dataset_meta['classes']

        # summary table
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
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
        
        # provide per class results for logger hook
        metrics['PerClass'] = ret_metrics_class

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics



class Load_SA_Med2D(LoadBiomedicalImageFromFile):
    def __init__(self, load_type:set[str], *args, **kwargs):
        self.load_type = load_type
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_png(png_path:str):
        png_file = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if png_file.ndim == 2:
            png_file = png_file[..., np.newaxis]
        return png_file

    def transform(self, results:Dict) -> Dict:
        raise NotImplementedError



class Load_SA_Med2D_SingleSlice(Load_SA_Med2D):
    def transform(self, results:Dict) -> Dict:
        if 'image' in self.load_type:
            # H W C
            image = self.load_png(os.path.join(
                results['img_path'][0], 
                f"image_{results['img_path'][1]}.png"))
            results['img'] = image.astype(np.uint16)
            results['img_shape'] = image.shape[:2]
            results['ori_shape'] = image.shape[:2]
            
        if 'label' in self.load_type:
            # H W
            label = self.load_png(os.path.join(
                results['seg_map_path'][0], 
                f"mask_{results['seg_map_path'][1]}.png"))
            results['gt_seg_map'] = label
            results['reduce_zero_label'] = False
        
        return results



class Load_SA_Med2D_MultiSlices(Load_SA_Med2D):
    def __init__(self, multi_img_load:bool, lazy_load:bool=False, *args, **kwargs):
        self.multi_img_load = multi_img_load    # 此参数仅被设计用于区分Train or Val
        # Lazy Load will delay the actual loading until the pixel array
        # are actually needed by the following preprocessing steps.
        # While not fully initialized, the pixel array is will remain to be
        # the path str pointing to the image.
        # This is particularly useful when not all images will
        # be used after sampling as a sample.
        # Reduce CPU Loading Stress.
        # Need Support in the following preprocessing step.
        self.lazy_load = lazy_load
        super().__init__(*args, **kwargs) # 多样本采样时，默认不支持在Load阶段进行归一化


    def transform(self, results: Dict) -> Dict:
        assert isinstance(results['multi_img_path'], List) and isinstance(results['multi_seg_map_path'], List), \
            f"MultiSlice Load needs Multiple Paths, but got img {type(results['multi_img_path'])} and label {type(results['multi_seg_map_path'])}"

        # Load images if required
        if 'image' in self.load_type:
            img_cache = []
            if self.multi_img_load:
                for img_idx in results['multi_img_path']:
                    img_cache.append(self.load_png(os.path.join(
                                    results['img_path'][0], 
                                    f"image_{img_idx}.png")))
            else:   # 仅加载Axial Image
                img_cache.append(self.load_png(os.path.join(
                                    results['img_path'][0], 
                                    f"image_{results['img_path'][1]}.png")))
            
            img_cache = np.stack(img_cache, axis=0)
            S, H, W, C = img_cache.shape
            img_ndarray = np.array(img_cache).transpose(1,2,0,3).reshape(H,W,S*C) # (H,W,S*C)
            results['img'] = img_ndarray.astype(np.uint16)
            results['img']
        
        # Load labels if required
        if 'label' in self.load_type:
            label_cache = []
            if self.multi_img_load:
                for label_idx in results['multi_seg_map_path']:
                    label_cache.append(self.load_png(os.path.join(
                                    results['seg_map_path'][0],
                                    f"mask_{label_idx}.png")))
            else:   # 仅加载Axial Label
                label_cache.append(self.load_png(os.path.join(
                                results['seg_map_path'][0], 
                                f"mask_{results['seg_map_path'][1]}.png")))
            
            label_cache = np.stack([hwc.squeeze() for hwc in label_cache], axis=0)
            assert label_cache.ndim == 3, f"Label should be 3-dim, but got {label_cache.shape}"
            results['gt_seg_map'] = label_cache
            results['reduce_zero_label'] = False    # 不可以随意更改此参数，会引起mmseg中多个行为改变
        
        return results



class Normalize(BaseTransform):
    def __init__(self, mode:str, size:Tuple[int,int], **kwargs):
        self.mode = mode
        self.H, self.W = size
        self._set_mask()
        super().__init__(**kwargs)
    
    def _set_mask(self):
        mask = np.zeros((self.H, self.W), np.uint8)
        # 计算中心区域的起始坐标
        H_l, H_h = self.H//4*1, self.H//4*3
        W_l, W_h = self.W//4*1, self.W//4*3
        # 设置中心区域为1
        mask[H_l:H_h, W_l:W_h] = 1
        self.statistic_mask = mask
    
    def transform(self, results:Dict):
        # results['img']: (H, W, C)
        image:np.ndarray = results['img'].astype(np.float32)

        if self.mode == 'dataset':
            mean = np.array(results['mean'])[np.newaxis, np.newaxis, :]
            std = np.array(results['std'])[np.newaxis, np.newaxis, :]
            image = image / std # ignore mean calibration to prevent the background area offset to none zero value

        elif self.mode == 'instance':
            image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

        elif self.mode == 'roi':
            # image: [H,W,C]
            mean, std = cv2.meanStdDev(src=image, mask=self.statistic_mask)
            std = std.squeeze(-1)[np.newaxis, np.newaxis, ...]  # (1, 1, C)
            image = image / std
        
        elif self.mode is None:
            pass
        else:
            raise NotImplementedError
        
        results['img'] = image
        return results



class Normalize_MultiSlice(Normalize):
    def transform(self, results:Dict):
        # results['img']: (H, W, S*C) (After Optical Flow Augmentation)
        C = len(results['min']) # 统计时是按照通道数存储的，每个通道有各自的统计量
        H, W, SC = results['img'].shape
        S = SC // C
        image:np.ndarray = results['img'].reshape(H, W, S, C).astype(np.float32)

        if self.mode == 'dataset':
            mean = np.array(results['mean'])[np.newaxis, np.newaxis, np.newaxis, :]
            std = np.array(results['std'])[np.newaxis, np.newaxis, np.newaxis, :]
            image = (image - mean) / std
                    
        elif self.mode == 'instance':
            for s in range(S):
                image[:,:,s] = cv2.normalize(image[:,:,s], None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
        
        elif self.mode == 'roi':
            # image: [H,W,S,C]
            for s in range(S):
                # mean, std: [C,1]
                mean, std = cv2.meanStdDev(src=image[:,:,s], mask=self.statistic_mask)
                std = std.squeeze()[np.newaxis, np.newaxis, :]  # (1,1,C)
                image[:,:,s] = image[:,:,s] / std   # 为防止image的背景向负方向偏移，不对mean进行修正
        
        elif self.mode is None:
            pass
        else:
            raise NotImplementedError
        
        results['img'] = image.astype(np.float32).reshape(H, W, SC)
        return results



class ForceResize(BaseTransform):
    def __init__(self, image_size:Tuple[int, int]=None, label_size:Tuple[int, int]=None):   # type:ignore
        self.image_size = image_size
        self.label_size = label_size
        
    def transform(self, results:Dict):
        if self.image_size:
            # (H, W, C)
            results['img'] = cv2.resize(
                results['img'], 
                dsize=self.image_size,
                interpolation=cv2.INTER_NEAREST)
            # 当通道数仅为1时，cv2.resize会把通道维度删除
            if results['img'].ndim == 2:
                results['img'] = results['img'][..., np.newaxis]    # (H, W) -> (H, W, C)
            results['img_shape'] = self.image_size[:2]
        
        if self.label_size:
            # (H, W)
            results['gt_seg_map'] = cv2.resize(
                results['gt_seg_map'].squeeze(), 
                dsize=self.label_size,
                interpolation=cv2.INTER_NEAREST)[np.newaxis,...]
        
        results['img_shape'] = self.image_size
        results['ori_shape'] = self.label_size
        return results



class LabelSqueeze(BaseTransform):
    def transform(self, results):
        results['gt_seg_map'] = results['gt_seg_map'].squeeze()
        return results



class ClassFilter(BaseTransform):
    def __init__(self, max_label_index:int):
        self.max_label_index = max_label_index

    def transform(self, results):
        gt_seg_map = results['gt_seg_map']
        gt_seg_map[gt_seg_map > self.max_label_index] = 0
        results['gt_seg_map'] = gt_seg_map
        return results



class ClassRectify(BaseTransform):
    def __init__(self):
        proxy:Type = DatasetBackend_GlobalProxy.get_current_instance()
        self.max_cls_idx = len(proxy.atom_classes) - 1
        self.union_atom_map = proxy.union_atom_map
        # print_log(f"Class Info Check: MaxClsIdx {self.max_cls_idx} | UnionAtomMap {self.union_atom_map}", 
        #           logger=MMLogger.get_current_instance())
    
    def transform(self, results):
        try:
            seg_map = results['gt_seg_map']
            for i in np.unique(seg_map):
                new_idx = self.union_atom_map.get(str(i), None)
                if new_idx:
                    seg_map[seg_map==int(i)] = new_idx
            if seg_map.max() > self.max_cls_idx:
                raise ValueError(f"Class index {seg_map.max()} is larger than valid maximum {self.max_cls_idx}")
            results['gt_seg_map'] = seg_map
        except ValueError as e:
            print(Fore.RED 
                  + f"\n\nEncounter exception value after ClassRectify: {seg_map.max()},"
                    f"expected max value: {self.max_cls_idx}\n" 
                    f"PROXY: {proxy.__dict__}\n\n"
                  + Style.RESET_ALL)
            exit(-2)
            
        return results



