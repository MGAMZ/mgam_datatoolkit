import os
import pdb
import os.path as osp
from typing_extensions import Sequence


import torch
import numpy as np

from mmseg.structures import SegDataSample
from mmseg.engine.hooks import SegVisualizationHook
from mmengine.runner import Runner

from mmengine import ConfigDict
from mmcv.transforms.base import BaseTransform

from .lmdb_GastricCancer import LMDB_DataBackend, LMDB_MP_Proxy





class LoadCTImage(BaseTransform):
    def __init__(self, lmdb_backend_proxy) -> None:
        super().__init__()
        if lmdb_backend_proxy:
            lmdb_backend_proxy = LMDB_MP_Proxy.get_current_instance()
            self.lmdb_service:LMDB_DataBackend = lmdb_backend_proxy()

    def transform(self, results: dict) -> dict:
        if isinstance(results['img_path'], np.ndarray):
            img = results['img_path']
            if hasattr(self, 'lmdb_service'):
                (path, meta_buffer, img_buffer) = self.lmdb_service.fetch_data(results['img_path']) # type.ignore
                (meta, _) = LMDB_DataBackend.decompress(meta_buffer, None)
                results['dcm_meta'] = meta
        else:
            if hasattr(self, 'lmdb_service'):
                (path, meta_buffer, img_buffer) = self.lmdb_service.fetch_data(results['img_path'])
                (meta, img) = LMDB_DataBackend.decompress(meta_buffer, img_buffer)
                results['dcm_meta'] = meta
            else:
                img = np.load(results['img_path'])

        results['img'] = img.astype(np.float32).squeeze()  # type:ignore
        results['img_shape'] = img.shape # type:ignore
        results['ori_shape'] = img.shape # type:ignore
        return results	# img: [H,W,C]


class LoadCTLabel(BaseTransform):
    def __init__(self, lmdb_backend_proxy) -> None:
        super().__init__()
        if lmdb_backend_proxy:
            assert isinstance(lmdb_backend_proxy, ConfigDict), "lmdb_backend_proxy must be a ConfigDict"
            lmdb_backend_proxy = LMDB_MP_Proxy.get_current_instance()
            self.lmdb_service:LMDB_DataBackend = lmdb_backend_proxy()
            self.EmptyLabel = np.zeros((512,512), dtype=np.uint8)

    def transform(self, results: dict) -> dict:
        if results['seg_map_path'] is None:
            gt_seg_map = self.EmptyLabel
        else:
            if hasattr(self, 'lmdb_service'):
                (path, meta_buffer, img_buffer) = self.lmdb_service.fetch_data(results['seg_map_path'])
                (meta, gt_seg_map) = LMDB_DataBackend.decompress(meta_buffer, img_buffer)
            else:
                gt_seg_map = np.load(results['seg_map_path'])
        
        results['gt_seg_map'] = gt_seg_map.astype(np.uint8).squeeze()   # type:ignore
        results['seg_fields'].append('gt_seg_map')
        return results	# gt_seg_map: [H,W]


class CTSegVisualizationHook(SegVisualizationHook):
    def __init__ (self, reverse_stretch=None, lmdb_backend_proxy=None, **kwargs):
        super().__init__(**kwargs)

        if lmdb_backend_proxy:
            lmdb_backend_proxy = LMDB_MP_Proxy.get_instance('LMDB_MP_Proxy', lmdb_args=lmdb_backend_proxy)
            self.lmdb_service:LMDB_DataBackend = lmdb_backend_proxy()


    def source_img(self, output:SegDataSample):
        if hasattr(self, 'lmdb_service'):
            (path, meta_buffer, img_buffer) = self.lmdb_service.fetch_data(output.img_path)
            (meta, pixel) = LMDB_DataBackend.decompress(meta_buffer, img_buffer)
        else:
            pixel = np.load(output.img_path)
        
        return (np.clip(pixel.reshape(512,512,1),0,4095)//16).astype(np.uint8)


    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """

        if self.draw is False or mode == 'train':
            return

        # mmseg visualization
        if self.every_n_inner_iters(batch_idx, self.interval):
            # 重载原始图像
            img_list = []
            for output in outputs:
                img_list.append(self.source_img(output))
            
            if hasattr(self, 'reverse_stretch'):
                # 推理预测 反拉伸
                original_pred_sem_seg_shape = outputs[0].pred_sem_seg.data.shape
                reverse_stretch_queue = [output.pred_sem_seg.data.squeeze().cpu().to(torch.uint8) 
                                         for output in outputs]
                
                # 执行多进程反拉伸并解包
                sample_iterator = self.reverse_stretch.multiprocess_stretch(reverse_stretch_queue)
                assert len(sample_iterator) == len(outputs)
                for batch_index in range(len(outputs)):
                    outputs[batch_index].pred_sem_seg.data = \
                        sample_iterator[batch_index].reshape(*original_pred_sem_seg_shape)

            for i, output in enumerate(outputs):
                img_path = output.img_path
                img = img_list[i]
                
                window_name = f'{mode}_{osp.basename(img_path)}'
                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)

