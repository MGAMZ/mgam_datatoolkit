import pdb
import logging
from collections.abc import Sequence

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from mmcv.transforms import BaseTransform
from mmengine.logging import print_log, MMLogger
from mmengine.evaluator.metric import BaseMetric
from mmengine.structures import PixelData
from mmengine.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.models.segmentors import EncoderDecoder

from ..mm.visualization import BaseViser, master_only, BaseDataElement
from ..mm.mgam_models import mgam_Seg2D_Lite



class AccuCount(BaseMetric):
    """
    Designed for counting the number of cells in a given image.
    The counts is calculated by directly add all pixels together.
    """

    def __init__(
        self,
        amplify: float = 1.0,
        low_high_threshold: int = 10,  # 细胞数量阈值，默认为10个
        low_abs_error: int = 3,  # 如patch中细胞数量<阈值, 绝对误差不超过3个
        high_rel_error: float = 0.3,  # 如patch中细胞数量>=阈值, 相对误差<=30%
        eps = 1e-7,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.amplify = amplify
        self.low_high_threshold = low_high_threshold
        self.low_abs_error = low_abs_error
        self.high_rel_error = high_rel_error
        self.eps = eps

    def _patchwise_product_met_rate(self, pred: float, label: float):
        """
        1. 如patch中细胞数量<10, 误差不超过3个
        2. 如patch中细胞数量>=10, 误差<=30%
        """
        if label < self.low_high_threshold:
            return np.abs(pred - label) <= self.low_abs_error
        elif label >= self.low_high_threshold:
            return (np.abs(pred - label) / (label+self.eps)) <= self.high_rel_error
        else:
            raise RuntimeError(f"Unknown Exception, pred: {pred}, label: {label}.")

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred = data_sample["seg_logits"]["data"].sum().cpu().numpy()
            label = data_sample["gt_sem_seg"]["data"].sum().cpu().numpy() / self.amplify
            product_met = self._patchwise_product_met_rate(pred, label)
            self.results.append(
                {"pred_count": pred, 
                 "gt_count": label, 
                 "product_met": product_met}
            )

    def compute_metrics(self, results: list[dict[str, np.ndarray]]) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        total_count = np.sum([res["gt_count"] for res in results])
        if total_count <= 1:
            print_log(f"GT Count too low ({total_count}), there may be some error?", logger=MMLogger.get_current_instance(), level=logging.WARN)
        return {
            "mae": np.mean(
                [np.abs(res["pred_count"] - res["gt_count"]) for res in results]
            ),
            "mape": np.mean(
                [
                    np.abs(res["pred_count"] - res["gt_count"]) / res["gt_count"]
                    for res in results
                ]
            ),
            "product_met": np.mean([res["product_met"] for res in results]),
        }


class HeatMapDownSample(BaseTransform):
    def __init__(self, ratio):
        self.ratio = ratio
    
    def transform(self, results:dict):
        if "gt_seg_map" in results:
            img = results["gt_seg_map"]
            img = cv2.resize(img, (img.shape[1]//self.ratio, img.shape[0]//self.ratio))
            results["gt_seg_map"] = img * (self.ratio**2)
        return results


class CellCounter(EncoderDecoder):
    def __init__(self, amplify:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amplify = amplify

    def postprocess_result(self, seg_logits, data_samples:Sequence[SegDataSample]|None=None):
        """Delete post-process sigmoid activation when C=1"""
        B, C, H, W = seg_logits.shape
        seg_logits = seg_logits / self.amplify
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(B)]
        for i, i_seg_logits in enumerate(seg_logits):
            data_samples[i].set_data({"seg_logits": PixelData(data=i_seg_logits),
                                      "pred_sem_seg": PixelData(data=i_seg_logits)})
        return data_samples


class CellCounterLite(mgam_Seg2D_Lite):
    def __init__(self, amplify:int, *args, **kwargs):
        super().__init__(num_classes=1, *args, **kwargs)
        self.amplify = amplify
    
    def predict(self, inputs:Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Sequence[BaseDataElement]:
        seg_logits = self.inference(inputs, data_samples) # [N, C, H, W]
        batch_size = inputs.shape[0]
        if data_samples is None:
            data_samples = [BaseDataElement() for _ in range(batch_size)]
        for i in range(batch_size):
            data_samples[i].seg_logits = PixelData(data=seg_logits[i])
        return data_samples
    
    @torch.inference_mode()
    def inference(self, inputs: Tensor, data_samples:Sequence[BaseDataElement]|None=None) -> Tensor:
        seg_logits = super().inference(inputs, data_samples)
        seg_logits /= self.amplify
        return seg_logits


class CellCounterClassifier(CellCounter):
    def __init__(self, amplify, ClasterClassifier, *args, **kwargs):
        super().__init__(amplify=amplify, *args, **kwargs)
        self.claster_classifier = MODELS.build(ClasterClassifier)


class Normalizer_cell2(BaseTransform):
    # RGB order
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def transform(self, results:dict):
        results['img'] = (results['img']/255 - self.mean) / self.std
        return results


class BGR2RGB(BaseTransform):
    def transform(self, results:dict):
        results['img'] = results['img'][..., ::-1]
        return results


class HeatMapViser(BaseViser):
    def __init__(self,
                 name:str="RoseThyroidCellCount_HeatMapViser",
                 alpha:float=0.6,
                 gt_amplify:float=1.,
                 *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.alpha = alpha
        self.gt_amplify = gt_amplify

    def _draw_heatmap(
        self,
        image: np.ndarray,
        gt_seg: BaseDataElement,
        seg_logit: BaseDataElement,
    ) -> np.ndarray:
        gt_seg_array = gt_seg.data.squeeze().cpu().numpy() / self.gt_amplify
        seg_logit_array = seg_logit.data.squeeze().cpu().numpy()
        
        assert (gt_seg_array.shape == seg_logit_array.shape), \
            f"Shape mismatch: gt_seg_array {gt_seg_array.shape} != sem_seg_array {seg_logit_array.shape}"
        if image.shape != gt_seg_array.shape:
            resize_ratio = (image.shape[0] / gt_seg_array.shape[0], image.shape[1] / gt_seg_array.shape[1])
            gt_seg_array = cv2.resize(gt_seg_array, image.shape[:-1], interpolation=cv2.INTER_NEAREST)
            gt_seg_array = gt_seg_array / resize_ratio[0] / resize_ratio[1]
        assert (image.shape[:2] == gt_seg_array.shape[:2]), \
            f"Shape mismatch: image {image.shape[:2]} != gt_seg_array {gt_seg_array.shape[:2]}"

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # draw gt
        axes[0].set_title("Ground Truth")
        axes[0].imshow(image)
        p1 = axes[0].imshow(gt_seg_array, alpha=self.alpha, cmap="hot")
        axes[0].text(
            0.1,
            0.5,
            f"Mask Info: "
            f"\nmean:{gt_seg_array.mean():.5f}\nstd:{gt_seg_array.std():.5f}"
            f"\nmax:{gt_seg_array.max():.5f}\nmin:{gt_seg_array.min():.5f}"
            f"\nsum:{gt_seg_array.sum():.5f}",
            fontsize=12,
            color="black",
            transform=axes[0].transAxes,
        )
        fig.colorbar(p1, ax=axes[0])
        
        # draw pred
        axes[1].set_title("Prediction")
        axes[1].imshow(image)
        p2 = axes[1].imshow(seg_logit_array, alpha=self.alpha, cmap="hot")
        axes[1].text(
            0.1,
            0.5,
            f"Mask Info: "
            f"\nmean:{seg_logit_array.mean():.5f}\nstd:{seg_logit_array.std():.5f}"
            f"\nmax:{seg_logit_array.max():.5f}\nmin:{seg_logit_array.min():.5f}"
            f"\nsum:{seg_logit_array.sum():.5f}",
            fontsize=12,
            color="black",
            transform=axes[1].transAxes,
        )
        fig.colorbar(p2, ax=axes[1])

        fig.tight_layout()
        heatmap = self.export_fig_to_ndarray(fig)
        return heatmap

    @master_only
    def add_datasample(self,
                       name,
                       image: np.ndarray,
                       data_sample: BaseDataElement,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        try:
            sample_file = data_sample.get('img_path')
            image_arr = np.load(sample_file)['img']
            drown_array = self._draw_heatmap(image_arr, data_sample.gt_sem_seg, data_sample.seg_logits)
            self.add_image(name, drown_array, step)
        except Exception as e:
            import traceback
            print("在执行HeatMap可视化时发生错误: ", e)
            traceback.print_exc()