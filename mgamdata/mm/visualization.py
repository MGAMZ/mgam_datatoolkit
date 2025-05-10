import logging
import os
import pdb
from abc import abstractmethod
from typing_extensions import Sequence

import cv2
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.logging import print_log, MMLogger
from mmengine.visualization.visualizer import Visualizer, master_only, BaseDataElement
from mmengine.visualization.vis_backend import LocalVisBackend as _LocalVisBackend
from mmengine.visualization.vis_backend import TensorboardVisBackend as _TensorboardVisBackend



class LocalVisBackend(_LocalVisBackend):
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        assert image.dtype == np.uint8
        drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(self._img_save_dir, exist_ok=True)
        save_file_name = f'{name}_{step}.png'.replace('/', '__') # support working with tensorboard tag rule.
        cv2.imwrite(os.path.join(self._img_save_dir, save_file_name), drawn_image)


class mgam_TensorboardVisBackend(_TensorboardVisBackend):
    def add_image(self, *args, **kwargs):
        super().add_image(*args, **kwargs)
        self._tensorboard.flush()


class BaseVisHook(Hook):
    def __init__(self, 
                 enabled:bool=True, 
                 val_vis_interval:int=50,
                 test_vis_interval:int=5):
        self._visualizer:Visualizer = Visualizer.get_current_instance()
        self.enabled = enabled
        self.val_vis_interval = val_vis_interval
        self.test_vis_interval = test_vis_interval
    
    def after_val_iter(self, 
                       runner: Runner, 
                       batch_idx: int, 
                       data_batch: dict,
                       outputs: Sequence[BaseDataElement]) -> None:
        for i in range(len(outputs)):
            if batch_idx % self.val_vis_interval == 0 and self.enabled:
                window_name = f'ValVis/Batch{batch_idx}Item{i}_{os.path.basename(outputs[i].img_path)}'
                self._visualizer.add_datasample(
                    window_name,
                    data_batch['inputs'][i], # dict with keys 'inputs' and 'data_samples'
                    data_sample=outputs[i],
                    step=runner.iter)

    def after_test_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: dict,
                        outputs: Sequence[BaseDataElement]) -> None:
        for i in range(len(outputs)):
            if batch_idx % self.test_vis_interval == 0 and self.enabled:
                window_name = f'TestVis/Batch{batch_idx}Item{i}_{os.path.basename(outputs[i].img_path)}'
                self._visualizer.add_datasample(
                    window_name,
                    data_batch['inputs'][i],
                    data_sample=outputs[i],
                    step=0)


class BaseViser(Visualizer):
    def export_fig_to_ndarray(self, fig:Figure, close:bool=True):
        try:
            fig.canvas.draw()
            
            if hasattr(fig.canvas, 'tostring_argb'):
                data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                # Properly handle ARGB format by rearranging channels
                width, height = fig.canvas.get_width_height()
                data = data.reshape((height, width, 4))
                # Convert ARGB to RGB by dropping alpha or handling it
                data = data[:, :, 1:4]  # Skip alpha channel (first channel)
                return data
            elif hasattr(fig.canvas, 'tostring_rgb'):
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                channels = 3  # RGB
            elif hasattr(fig.canvas, 'buffer_rgba'):
                data = np.asarray(fig.canvas.buffer_rgba())
                channels = 4  # RGBA
            else:
                raise ValueError("Unknown canvas type - canvas methods not supported")
                
            width, height = fig.canvas.get_width_height()
            data = data.reshape((height, width, channels))
            
            return data
        
        finally:
            if close:
                plt.close(fig)

    @master_only
    @abstractmethod
    def add_datasample(self,
                       name,
                       image: np.ndarray,
                       data_sample: BaseDataElement|None = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        """Draw datasample."""


class SegViser(BaseViser):
    def __init__(self, 
                 name:str='SegViser',
                 dim:int=2,
                 gt_seg_key:str='gt_sem_seg',
                 pred_seg_key:str='pred_sem_seg',
                 pred_seg_logits_key:str|None='seg_logits',
                 sem_seg_map_min_size:int|Sequence[int]|None=None,
                 image_cmap:str='gray',
                 seg_map_cmap:str='rainbow',
                 seg_map_alpha:float=0.3,
                 plt_figsize:tuple[int, int]=(15, 4),
                 plt_invert:bool=False,
                 verbose:bool=False,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.verbose = verbose
        
        self.gt_seg_key = gt_seg_key
        self.pred_seg_key = pred_seg_key
        self.pred_seg_logits_key = pred_seg_logits_key
        self.sem_seg_map_min_size = sem_seg_map_min_size
        
        self.image_cmap = image_cmap
        self.seg_map_cmap = seg_map_cmap
        self.seg_map_alpha = seg_map_alpha
        self.plt_figsize = plt_figsize
        self.plt_invert = plt_invert

    def _parse_datasample(self, 
                         image:np.ndarray|torch.Tensor,
                         data_sample:BaseDataElement|None=None
                         ) -> tuple[np.ndarray|None, np.ndarray|None, np.ndarray|None, np.ndarray|None]:
        """Parse data sample.
        
        Arg:
            - image: np.ndarray|None: Its shape can be (C,[Z],Y,X)
            - data_sample: BaseDataElement|None
                - gt_seg_map: np.ndarray|None: Its shape can be ([Z],Y,X)
                - pred_seg_map: np.ndarray|None: Its shape can be ([Z],Y,X)
                - pred_seg_logits: np.ndarray|None: Its shape can be (C,[Z],Y,X)
        
        Return:
            - image: np.ndarray|None: Its shape will be (Y,X,C)
            - gt_seg_map: np.ndarray|None: Its shape will be (Y,X)
            - pred_seg_map: np.ndarray|None: Its shape will be (Y,X)
            - pred_seg_logits: np.ndarray|None: Its shape will be (Y,X,C)
        
        """
        
        def to_ndarray(key):
            data = data_sample.get(key, None)
            if isinstance(data, BaseDataElement):
                data = data.data
            
            if isinstance (data, torch.Tensor):
                return data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                raise NotImplementedError(f"Unsupported data type when visualizing: {type(data)}")
        
        image = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        gt_seg_map = to_ndarray(self.gt_seg_key)
        pred_seg_map = to_ndarray(self.pred_seg_key)
        pred_seg_logits = to_ndarray(self.pred_seg_logits_key)
        if gt_seg_map is not None:
            gt_seg_map = gt_seg_map.squeeze() # (1,Z,Y,X) -> (Z,Y,X)
        if pred_seg_map is not None:
            pred_seg_map = pred_seg_map.squeeze() # (1,Z,Y,X) -> (Z,Y,X)
        
        # When image is Volume, Z must be determined.
        if self.dim == 3:
            def find_foreground(array):
                Z = np.where(np.any(array, axis=(1,2)))[0]
                return None if len(Z) == 0 else Z[len(Z)//2]
            
            # The Z containing foreground will be the priority.
            if gt_seg_map is not None:
                Z = find_foreground(gt_seg_map)
            if Z is None and pred_seg_map is not None:
                Z = find_foreground(pred_seg_map)
            if Z is None and pred_seg_logits is not None:
                Z = find_foreground(pred_seg_logits.any(axis=0)) # C,Z,Y,X -> Z,Y,X
            # If there is no foreground, Z will be randomly selected.
            if Z is None:
                Z = image.shape[1] // 2
        
            # Slice the Volume
            image = image[:, Z]
            if gt_seg_map is not None:
                gt_seg_map = gt_seg_map[Z]
            if pred_seg_map is not None:
                pred_seg_map = pred_seg_map[Z]
            if pred_seg_logits is not None:
                pred_seg_logits = pred_seg_logits[:, Z]
        
        # move channel to the last dimension
        if image is not None:
            image = image.transpose(1, 2, 0)
        if pred_seg_logits is not None:
            pred_seg_logits = pred_seg_logits.transpose(1, 2, 0)
        
        # Their shape will be ensured as: (Y,X), (Y,X), (Y,X), (Y,X,C)
        return image, gt_seg_map, pred_seg_map, pred_seg_logits

    def _draw_fig(self,
                  img_path:str,
                  image:np.ndarray|None,
                  gt_seg_map:np.ndarray|None,
                  pred_seg_map:np.ndarray|None,
                  pred_seg_logits:np.ndarray|None
    ):
        fig, axes = plt.subplots(1, 4, figsize=self.plt_figsize)
        fig.suptitle(img_path, fontsize=9)
        
        # Draw image (Y,X,C)
        if image is not None:
            axes[0].imshow(image, cmap=self.image_cmap, interpolation='bicubic')
            axes[0].set_title('image')
        else:
            axes[0].set_title('image N/A')
        
        # Draw gt_seg_map (Y,X)
        if gt_seg_map is not None:
            gt_seg_map = gt_seg_map.copy().astype(float)
            gt_seg_map[gt_seg_map == 0] = np.nan
            axes[1].imshow(image, cmap=self.image_cmap, interpolation='bicubic')
            axes[1].imshow(gt_seg_map, cmap=self.seg_map_cmap, interpolation='nearest', alpha=self.seg_map_alpha)
            axes[1].set_title('gt_seg_map')
        else:
            axes[1].set_title('gt_seg_map N/A')
        
        # Draw pred_seg_map (Y,X)
        if pred_seg_map is not None:
            pred_seg_map = pred_seg_map.copy().astype(float)
            pred_seg_map[pred_seg_map == 0] = np.nan
            axes[2].imshow(image, cmap=self.image_cmap, interpolation='bicubic')
            axes[2].imshow(pred_seg_map, cmap=self.seg_map_cmap, interpolation='nearest', alpha=self.seg_map_alpha)
            axes[2].set_title('pred_seg_map')
        else:
            axes[2].set_title('pred_seg_map N/A')
        
        # Draw pred_seg_logits (Y,X,C)
        # calculate confidence
        if pred_seg_logits is not None:
            axes[3].imshow(image, cmap=self.image_cmap, interpolation='bicubic')
            logits_max = np.max(pred_seg_logits, axis=-1, keepdims=True) # prevent overflow
            exp_logits = np.exp(pred_seg_logits - logits_max)
            probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-5)
            confidence = np.max(probs, axis=-1)
            im = axes[3].imshow(confidence, cmap='jet', vmin=0, vmax=1, alpha=self.seg_map_alpha)
            axes[3].set_title('confidence')
            plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        else:
            axes[3].set_title('confidence (pred_seg_logits) N/A')
        
        if self.plt_invert:
            for ax in axes:
                ax.invert_yaxis()
                ax.invert_xaxis()
        
        # Format
        fig.tight_layout()
        return self.export_fig_to_ndarray(fig)

    @master_only
    def add_datasample(self,
                       name:str,
                       image:np.ndarray|torch.Tensor,
                       data_sample:BaseDataElement|None=None,
                       step:int=0) -> None:
        if self.verbose:
            print_log(f"Visualizing `{name}` | Step {step} | "
                      f"image shape {image.shape if image is not None else None} | "
                      f"data sample keys: {data_sample.keys() if data_sample is not None else None}", 
                      MMLogger.get_current_instance(), logging.INFO)
        
        # parse datasample
        img_path = data_sample.metainfo.get('img_path', None)
        image, gt_seg_map, pred_seg_map, pred_seg_logits = self._parse_datasample(image, data_sample)
        if gt_seg_map is None:
            print_log(f"When visualizing `{name}` with img_path `{img_path}`, "
                      "gt_seg_map is None. So the gt_seg_map will not be empty.",
                      MMLogger.get_current_instance(), logging.WARN)
        
        # draw fig and save
        image_array = self._draw_fig(img_path, image, gt_seg_map, pred_seg_map, pred_seg_logits)
        self.add_image(name, image_array, step)
