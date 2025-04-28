import os.path as osp
import pdb
import warnings
from abc import abstractmethod
from prettytable import PrettyTable
from collections import OrderedDict
from typing_extensions import deprecated

import cv2
import torch
import numpy as np
from skimage.exposure import equalize_hist
from matplotlib import pyplot as plt

import mmcv
import mmengine
from mmengine.structures import PixelData
from mmengine.dist.utils import master_only
from mmengine.logging import print_log, MMLogger
from mmengine.runner import Runner
from mmseg.evaluation.metrics import IoUMetric
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.visualization import SegLocalVisualizer
from mmseg.structures import SegDataSample
from mmcv.transforms import BaseTransform, to_tensor


class HistogramEqualization(BaseTransform):
    def __init__(self, image_size: tuple, ratio: float):
        assert image_size[0] == image_size[1], "Only support square shape for now."
        assert ratio < 1, "RoI out of bounds"
        self.RoI = self.create_circle_in_square(image_size[0], image_size[0] * ratio)
        self.nbins = image_size[0]

    @staticmethod
    def create_circle_in_square(size: int, radius: int) -> np.ndarray:
        # 创建一个全0的正方形ndarray
        square = np.zeros((size, size))
        # 计算中心点的坐标
        center = size // 2
        # 计算每个元素到中心的距离
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
        # 如果距离小于或等于半径，将该元素设置为1
        square[mask] = 1
        return square

    def RoI_HistEqual(self, image: np.ndarray):
        dtype_range = np.iinfo(image)
        normed_image = equalize_hist(image, nbins=self.nbins, mask=self.RoI)
        normed_image = (normed_image * dtype_range.max).astype(image.dtype)
        return normed_image

    def transform(self, results: dict) -> dict:
        assert isinstance(results["img"], list)
        for i, image in enumerate(results["img"]):
            results["img"][i] = self.RoI_HistEqual(image)
        return results


class IoUMetric_PerClass(IoUMetric):
    def __init__(self, iou_metrics: list[str]=['mIoU', 'mDice', 'mFscore'], *args, **kwargs):
        super().__init__(iou_metrics=iou_metrics, *args, **kwargs)
    
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
            logger.info(f"results are saved to {osp.dirname(self.output_dir)}")
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect: torch.Tensor = sum(results[0])
        total_area_union: torch.Tensor = sum(results[1])
        total_area_pred_label: torch.Tensor = sum(results[2])
        total_area_label: torch.Tensor = sum(results[3])

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )
        class_names = self.dataset_meta["classes"]  # type: ignore

        # class averaged table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[key] = val
            else:
                metrics["m" + key] = val

        # each class table
        ret_metrics.pop("aAcc", None)
        class_metrics = OrderedDict(
            {
                ret_metric: [format(v, ".2f") for v in ret_metric_value * 100]
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        class_metrics.update({"Class": class_names})
        class_metrics.move_to_end("Class", last=False)
        class_table_data = PrettyTable()
        for key, val in class_metrics.items():
            class_table_data.add_column(key, val)

        # provide per class results for logger hook
        metrics["PerClass"] = class_metrics

        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)

        return metrics

@deprecated("mmseg_PlugIn's Vis module has been deprecated, please use `mgamdata.mm.visualization` instead.")
class SegVisualizationHook_Base(SegVisualizationHook):
    @abstractmethod
    def _get_source_image(self, data_sample: SegDataSample) -> np.ndarray: ...

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: list[SegDataSample],
    ) -> None:
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
        window_name = f"val_{osp.basename(outputs[0].img_path)}"
        img = self._get_source_image(outputs[0])
        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                step=total_curr_iter,
            )

    def after_test_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: list[SegDataSample],
    ) -> None:
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
            window_name = f"test_{osp.basename(data_sample.img_path)}"

            img = self._get_source_image(data_sample)
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index,
            )

@deprecated("mmseg_PlugIn's Vis module has been deprecated, please use `mgamdata.mm.visualization` instead.")
class SegViser(SegLocalVisualizer):
    def __init__(
        self,
        name,
        draw_heatmap: bool = False,
        draw_others: bool = False,
        amplify: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        self.draw_heatmap = draw_heatmap
        self.draw_others = draw_others
        self.amplify = amplify

    def _draw_heatmap(
        self,
        image: np.ndarray,
        gt_seg: mmengine.structures.PixelData,
        seg_logit: mmengine.structures.PixelData,
    ) -> np.ndarray:
        gt_seg_array = gt_seg.data.squeeze().cpu().numpy() / self.amplify
        seg_logit_array = seg_logit.data.squeeze().cpu().numpy() / self.amplify
        
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
        axes[0].imshow(image, cmap="gray")
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
        axes[1].imshow(image, cmap="gray")
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
        fig.canvas.draw()
        heatmap = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        heatmap = heatmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return heatmap

    @master_only
    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: SegDataSample | None = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        # TODO: Supported in mmengine's Viusalizer.
        out_file: str | None = None,
        step: int = 0,
        with_labels: bool | None = True,
    ) -> None:

        if self.draw_heatmap:
            heatmap = self._draw_heatmap(image, data_sample.gt_sem_seg, data_sample.seg_logits)
            self.add_image("heatmap_" + name, heatmap, step)
        if self.draw_others:
            super().add_datasample(
                name,
                image,
                data_sample,
                draw_gt,
                draw_pred,
                show,
                wait_time,
                out_file,
                step,
                with_labels,
            )

@deprecated("mmseg_PlugIn's Vis module has been deprecated, please use `mgamdata.mm.visualization` instead.")
class SegVisHook_Vanilla(SegVisualizationHook_Base):
    def _get_source_image(self, data_sample: SegDataSample) -> np.ndarray:
        img_path = data_sample.img_path
        img_bytes = mmengine.fileio.get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order="rgb")
        return img

@deprecated("mmseg_PlugIn's Vis module has been deprecated, please use `mgamdata.mm.visualization` instead.")
class SegVisHook_Npz(SegVisualizationHook_Base):
    def _get_source_image(self, data_sample: SegDataSample) -> np.ndarray:
        img = np.load(data_sample.img_path)["img"]
        return img


class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(
        self,
        meta_keys=(
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "reduce_zero_label",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results["inputs"] = img

        data_sample = SegDataSample()
        if "gt_seg_map" in results:
            if len(results["gt_seg_map"].shape) == 2:
                data = to_tensor(results["gt_seg_map"][None, ...])
            else:
                warnings.warn(
                    "Please pay attention your ground truth "
                    "segmentation map, usually the segmentation "
                    "map is 2D, but got "
                    f'{results["gt_seg_map"].shape}'
                )
                data = to_tensor(results["gt_seg_map"])
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str
