import os
import pdb
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mmcv.transforms import to_tensor
from mmengine.runner import Runner
from mmengine.structures.base_data_element import BaseDataElement
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.datasets.transforms import PackSegInputs
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses.accuracy import accuracy
from mmseg.visualization.local_visualizer import SegLocalVisualizer
from mmseg.structures.seg_data_sample import SegDataSample, PixelData

from ..criterions.segment import DiceLoss_3D



class VolumeData(BaseDataElement):
    """Data structure for volume-level annotations or predictions.

    All data items in ``data_fields`` of ``VolumeData`` meet the following
    requirements:

    - They all have 4 dimensions in orders of channel, Z, Y, and X.
    - They should have the same Z, Y, and X dimensions.

    Examples:
        >>> metainfo = dict(
        ...     volume_id=random.randint(0, 100),
        ...     volume_shape=(random.randint(20, 40), random.randint(400, 600), random.randint(400, 600)))
        >>> volume = np.random.randint(0, 255, (4, 30, 20, 40))
        >>> featmap = torch.randint(0, 255, (10, 30, 20, 40))
        >>> volume_data = VolumeData(metainfo=metainfo,
        ...                          volume=volume,
        ...                          featmap=featmap)
        >>> print(volume_data.shape)
        (30, 20, 40)

        >>> # slice
        >>> slice_data = volume_data[10:20, 5:15, 10:30]
        >>> assert slice_data.shape == (10, 10, 20)
        >>> slice_data = volume_data[10, 5, 10]
        >>> assert slice_data.shape == (1, 1, 1)

        >>> # set
        >>> volume_data.map3 = torch.randint(0, 255, (30, 20, 40))
        >>> assert tuple(volume_data.map3.shape) == (1, 30, 20, 40)
        >>> with self.assertRaises(AssertionError):
        ...     # The dimension must be 4 or 3
        ...     volume_data.map2 = torch.randint(0, 255, (1, 3, 30, 20, 40))
    """

    def __setattr__(self, name: str, value: Tensor | np.ndarray):
        """Set attributes of ``VolumeData``.

        If the dimension of value is 3 and its shape meet the demand, it
        will automatically expand its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `VolumeData`.
            value (Union[Tensor, np.ndarray]): The value to store in.
                The type of value must be `Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `VolumeData`.
        """
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f"{name} has been used as a private attribute, which is immutable.")

        else:
            assert isinstance(value, (Tensor, np.ndarray)), (
                f"Can not set {type(value)}, only support" f" {(Tensor, np.ndarray)}"
            )

            if self.shape:
                assert tuple(value.shape[-3:]) == self.shape, (
                    f"The Z, Y, and X dimensions of values {tuple(value.shape[-3:])} are "
                    f"not consistent with the shape of this :obj:`VolumeData` {self.shape}")
            assert value.ndim in [3,4,], f"The dim of value must be 3 or 4, but got {value.ndim}"
            if value.ndim == 3:
                value = value[None]
                warnings.warn(f"The shape of value will convert from {value.shape[-3:]} to {value.shape}")
            super().__setattr__(name, value)

    def __getitem__(self, item: Sequence[int | slice]) -> "VolumeData":
        """
        Args:
            item (Sequence[Union[int, slice]]): Get the corresponding values
                according to item.

        Returns:
            :obj:`VolumeData`: Corresponding values.
        """

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):
            assert len(item) == 3, "Only support to slice Z, Y, and X dimensions"
            tmp_item: list[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(0,slice(single_item, None, self.shape[-index - 1])) # type: ignore
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError(f"The type of element in input must be int or slice, but got {type(single_item)}")
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError(f"Unsupported type {type(item)} for slicing VolumeData")
        return new_data

    @property
    def shape(self):
        """The shape of volume data."""
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape[-3:])
        else:
            return None


class Seg3DDataSample(BaseDataElement):
    """A data structure interface of MMSegmentation for 3D data. They are used as
    interfaces between different components.

    The attributes in ``Seg3DDataSample`` are divided into several parts:

        - ``gt_sem_seg``(VolumeData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(VolumeData): Prediction of semantic segmentation.
        - ``seg_logits``(VolumeData): Predicted logits of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import VolumeData
         >>> from mmseg.structures import Seg3DDataSample

         >>> data_sample = Seg3DDataSample()
         >>> img_meta = dict(volume_shape=(4, 4, 4, 3),
         ...                 pad_shape=(4, 4, 4, 3))
         >>> gt_segmentations = VolumeData(metainfo=img_meta)
         >>> gt_segmentations.data = torch.randint(0, 2, (1, 4, 4, 4))
         >>> data_sample.gt_sem_seg = gt_segmentations
         >>> assert 'volume_shape' in data_sample.gt_sem_seg.metainfo_keys()
         >>> data_sample.gt_sem_seg.shape
         (4, 4, 4)
         >>> print(data_sample)
        <Seg3DDataSample(

            META INFORMATION

            DATA FIELDS
            gt_sem_seg: <VolumeData(

                    META INFORMATION
                    volume_shape: (4, 4, 4, 3)
                    pad_shape: (4, 4, 4, 3)

                    DATA FIELDS
                    data: tensor([[[[1, 1, 1, 0],
                                 [1, 0, 1, 1],
                                 [1, 1, 1, 1],
                                 [0, 1, 0, 1]]]])
                ) at 0x1c2b4156460>
        ) at 0x1c2aae44d60>

        >>> data_sample = Seg3DDataSample()
        >>> gt_sem_seg_data = dict(sem_seg=torch.rand(1, 4, 4, 4))
        >>> gt_sem_seg = VolumeData(**gt_sem_seg_data)
        >>> data_sample.gt_sem_seg = gt_sem_seg
        >>> assert 'gt_sem_seg' in data_sample
        >>> assert 'sem_seg' in data_sample.gt_sem_seg
    """

    @property
    def gt_sem_seg(self) -> VolumeData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: VolumeData) -> None:
        self.set_field(value, "_gt_sem_seg", dtype=VolumeData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def gt_sem_seg_one_hot(self) -> VolumeData:
        return self._gt_sem_seg_one_hot

    @gt_sem_seg_one_hot.setter
    def gt_sem_seg_one_hot(self, value: VolumeData) -> None:
        self.set_field(value, "_gt_sem_seg_one_hot", dtype=VolumeData)

    @gt_sem_seg_one_hot.deleter
    def gt_sem_seg_one_hot(self) -> None:
        del self._gt_sem_seg_one_hot

    @property
    def pred_sem_seg(self) -> VolumeData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: VolumeData) -> None:
        self.set_field(value, "_pred_sem_seg", dtype=VolumeData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg

    @property
    def seg_logits(self) -> VolumeData:
        return self._seg_logits

    @seg_logits.setter
    def seg_logits(self, value: VolumeData) -> None:
        self.set_field(value, "_seg_logits", dtype=VolumeData)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._seg_logits


class EncoderDecoder_3D(EncoderDecoder):
    """Encoder Decoder segmentors for 3D data."""

    def slide_inference(
        self,
        inputs: Tensor,
        batch_img_metas: list[dict],
    ) -> Tensor:
        """Inference by sliding-window with overlap.

        If z_crop > z_img or y_crop > y_img or x_crop > x_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxZxYxX,
                which contains all volumes in the batch.
            batch_img_metas (list[dict]): list of volume metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input volume.
        """

        accu_device: str = self.test_cfg.slide_accumulate_device
        z_stride, y_stride, x_stride = self.test_cfg.stride  # type: ignore
        z_crop, y_crop, x_crop = self.test_cfg.crop_size  # type: ignore
        batch_size, _, z_img, y_img, x_img = inputs.size()
        out_channels = self.out_channels
        z_grids = max(z_img - z_crop + z_stride - 1, 0) // z_stride + 1
        y_grids = max(y_img - y_crop + y_stride - 1, 0) // y_stride + 1
        x_grids = max(x_img - x_crop + x_stride - 1, 0) // x_stride + 1
        preds = torch.zeros(
            size=(batch_size, out_channels, z_img, y_img, x_img),
            dtype=torch.float16,
            device=accu_device,
            pin_memory=False,
        )
        count_mat = torch.zeros(
            size=(batch_size, 1, z_img, y_img, x_img),
            dtype=torch.uint8,
            device=accu_device,
            pin_memory=False,
        )

        for z_idx in range(z_grids):
            for y_idx in range(y_grids):
                for x_idx in range(x_grids):
                    z1 = z_idx * z_stride
                    y1 = y_idx * y_stride
                    x1 = x_idx * x_stride
                    z2 = min(z1 + z_crop, z_img)
                    y2 = min(y1 + y_crop, y_img)
                    x2 = min(x1 + x_crop, x_img)
                    z1 = max(z2 - z_crop, 0)
                    y1 = max(y2 - y_crop, 0)
                    x1 = max(x2 - x_crop, 0)
                    crop_vol = inputs[:, :, z1:z2, y1:y2, x1:x2]
                    # change the volume shape to patch shape
                    batch_img_metas[0]["img_shape"] = crop_vol.shape[2:]
                    # the output of encode_decode is seg logits tensor map
                    # with shape [N, C, Z, Y, X]
                    # NOTE WARNING:
                    # Setting `non_blocking=True` WILL CAUSE:
                    # Invalid pred_seg_logit accumulation on X axis.
                    crop_seg_logit = self.encode_decode(crop_vol, batch_img_metas).to(
                        accu_device, non_blocking=False
                    )
                    preds[:, :, z1:z2, y1:y2, x1:x2] += crop_seg_logit
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1

        assert torch.all(count_mat != 0), "The count_mat should not be zero"
        seg_logits = preds / count_mat
        return seg_logits

    def postprocess_result(
        self, seg_logits: Tensor, data_samples: list[Seg3DDataSample] | None = None
    ) -> list[Seg3DDataSample]:
        """Convert results list to `SegDataSample` for 3D Volume segmentation.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image with shape [B, C, Z, Y, X].
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(VolumeData): Prediction of semantic segmentation.
            - ``seg_logits``(VolumeData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, Z, Y, X = seg_logits.shape

        if data_samples is None:
            data_samples = [Seg3DDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if "img_padding_size" not in img_meta:
                    padding_size = img_meta.get("padding_size", [0] * 6)
                else:
                    padding_size = img_meta["img_padding_size"]
                (
                    padding_left,
                    padding_right,
                    padding_top,
                    padding_bottom,
                    padding_front,
                    padding_back,
                ) = padding_size
                # i_seg_logits shape is 1, C, Z, Y, X after remove padding
                i_seg_logits = seg_logits[
                    i : i + 1,
                    :,
                    padding_front : Z - padding_back,
                    padding_top : Y - padding_bottom,
                    padding_left : X - padding_right,
                ]

                flip = img_meta.get("flip", None)
                if flip:
                    flip_direction = img_meta.get("flip_direction", None)
                    assert flip_direction in ["horizontal", "vertical", "depth"]
                    if flip_direction == "horizontal":
                        i_seg_logits = i_seg_logits.flip(dims=(4,))
                    elif flip_direction == "vertical":
                        i_seg_logits = i_seg_logits.flip(dims=(3,))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2,))

                # resize as original shape
                i_seg_logits = F.interpolate(
                    i_seg_logits, size=img_meta["ori_shape"], mode="trilinear"
                ).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits > self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data(
                {
                    "seg_logits": VolumeData(**{"data": i_seg_logits}),  # type: ignore
                    "pred_sem_seg": VolumeData(**{"data": i_seg_pred}),  # type: ignore
                }
            )

        return data_samples


class BaseDecodeHead_3D(BaseDecodeHead):
    def __init__(
        self,
        loss_gt_key: str = "gt_sem_seg",
        deep_supervision_weight_truth: int = 2,
        loss_decode: dict = dict(type=DiceLoss_3D),
        *args,
        **kwargs,
    ):
        super().__init__(loss_decode=loss_decode, *args, **kwargs)
        self.loss_gt_key = loss_gt_key
        self.deep_supervision_weight_truth = deep_supervision_weight_truth
        self.conv_seg = torch.nn.Conv3d(self.channels, self.out_channels, kernel_size=1)
        if self.dropout_ratio > 0:
            self.dropout = torch.nn.Dropout3d(self.dropout_ratio)

    @abstractmethod
    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor]: ...

    def loss_per_layer(
        self,
        seg_logit: Tensor,
        seg_label: Tensor,
        losses_dict: dict,
        weight: float = 1.0,
    ) -> dict:
        seg_label = F.interpolate(
            input=seg_label,
            size=seg_logit.shape[2:],  # Skip batch and channel dimension.
            mode="nearest")

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, torch.nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in losses_dict:
                losses_dict[loss_decode.loss_name] = (
                    loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                    )
                    * weight
                )
            else:
                losses_dict[loss_decode.loss_name] += (
                    loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index,
                    )
                    * weight
                )

        return losses_dict

    def loss(
        self,
        inputs: tuple[Tensor],
        batch_data_samples: list[Seg3DDataSample],
        train_cfg:dict|None=None,
    ) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]):
                List of multi-level img features.
                (N, C, Z, Y, X)

            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.

            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        # list of Tensor: [B, C, Z, Y, X]
        seg_logits = self.forward(inputs)
        # [B, C, Z, Y, X]
        seg_label = self._stack_batch_gt(batch_data_samples, self.loss_gt_key)
        # HACK Deep Supervision Loss Calculation
        for i, seg_logit in enumerate(seg_logits):
            losses = self.loss_per_layer(
                seg_logit,
                seg_label,
                losses,
                weight=1 / (self.deep_supervision_weight_truth**i))

        losses["acc_seg"] = accuracy(seg_logits[0], 
                                     seg_label.squeeze(1), 
                                     ignore_index=self.ignore_index)

        return losses

    def predict(
        self, inputs: tuple[Tensor], batch_img_metas: list[dict], test_cfg: dict
    ) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        # Select the last output, shape: [B, C, Z, Y, X]
        seg_logits = self.forward(inputs)[0]

        if isinstance(batch_img_metas[0]["img_shape"], torch.Size):
            # slide inference
            size = batch_img_metas[0]["img_shape"]
        elif "pad_shape" in batch_img_metas[0]:
            size = batch_img_metas[0]["pad_shape"][:2]
        else:
            size = batch_img_metas[0]["img_shape"]

        seg_logits = F.interpolate(
            input=seg_logits,
            size=size,
            mode="nearest")
        return seg_logits

    def _stack_batch_gt(self, batch_data_samples: list[Seg3DDataSample], gt_key) -> Tensor:
        gt_semantic_segs = [data_sample.get(gt_key).data for data_sample in batch_data_samples]
        return torch.stack(gt_semantic_segs, dim=0)


class Seg3DVisualizationHook(SegVisualizationHook):
    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Sequence[Seg3DDataSample],
    ) -> None:
        if self.draw is False:
            return
        total_curr_iter = runner.iter + batch_idx

        # NOTE Override original implementation.
        # data batch inputs [N, C, Z, Y, X], but requires RGB at last dimension.
        img = data_batch["inputs"][0].permute(1, 2, 3, 0).numpy()
        img -= img.min()
        img /= img.max()
        img *= 255
        img = img.astype(np.uint8).copy()

        # img: [Z, Y, X, 1] -> [Z, Y, X, 3]
        img = np.repeat(img, 3, axis=-1)
        series_id = os.path.basename(outputs[0].metainfo["img_path"]).rstrip(".mha")
        window_name = f"val_{series_id}"

        if (total_curr_iter % self.interval == 0) or (total_curr_iter == 1):
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
        outputs: Sequence[Seg3DDataSample],
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

            # NOTE Override original implementation.
            # data batch inputs [N, C, Z, Y, X], but requires RGB at last dimension.
            img = data_batch["inputs"][0].permute(1, 2, 3, 0).numpy()
            # img: [Z, Y, X, 1] -> [Z, Y, X, 3]
            img = np.repeat(img, 3, axis=-1)
            series_id = os.path.basename(outputs[0].metainfo["img_path"]).rstrip(".mha")
            window_name = f"val_{series_id}"

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index,
            )


class Seg3DLocalVisualizer(SegLocalVisualizer):
    def __init__(
        self,
        name,
        resize: Sequence[int] | None = None,
        label_text_scale: float = 0.05,
        label_text_thick: float = 1,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        self.resize = resize
        self.label_text_scale = label_text_scale
        self.label_text_thick = label_text_thick

    def _draw_sem_seg(
        self,
        image: np.ndarray,
        sem_seg: PixelData,
        classes: list,
        palette: list,
        with_labels: bool = True,
    ) -> np.ndarray:
        "NOTE MGAM improve: configurable font size"

        num_classes = len(classes)

        sem_seg = sem_seg.cpu().data
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        colors = [palette[label] for label in labels]

        mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color

        if with_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # (0,1] to change the size of the text relative to the image
            scale = self.label_text_scale
            fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
            fontColor = (255, 255, 255)
            rectangleThickness = thickness = self.label_text_thick
            lineType = 2

            if isinstance(sem_seg[0], torch.Tensor):
                masks = sem_seg[0].numpy() == labels[:, None, None]
            else:
                masks = sem_seg[0] == labels[:, None, None]
            masks = masks.astype(np.uint8)
            for mask_num in range(len(labels)):
                classes_id = labels[mask_num]
                classes_color = colors[mask_num]
                loc = self._get_center_loc(masks[mask_num])
                text = classes[classes_id]
                (label_width, label_height), baseline = cv2.getTextSize(
                    text, font, fontScale, thickness
                )
                mask = cv2.rectangle(
                    mask,
                    loc,
                    (loc[0] + label_width + baseline, loc[1] + label_height + baseline),
                    classes_color,
                    -1,
                )
                mask = cv2.rectangle(
                    mask,
                    loc,
                    (loc[0] + label_width + baseline, loc[1] + label_height + baseline),
                    (0, 0, 0),
                    rectangleThickness,
                )
                mask = cv2.putText(
                    mask,
                    text,
                    (loc[0], loc[1] + label_height),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType,
                )
        color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(np.uint8)
        self.set_image(color_seg)
        return color_seg

    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Seg3DDataSample | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Randomly select a slice from the 3D volume and fall back to 2D visualize.

        Args:
            name: ...
            image (np.ndarray): The image to visualize, NdArray (Z, Y, X, C).
            data_sample (Seg3DDataSample, optional): The data sample to visualize.
                - gt_sem_seg (data:VolumeData): tensor (1, Z, Y, X)
                - pred_sem_seg (data:VolumeData): tensor (1, Z, Y, X)
                - seg_logits (data:VolumeData): tensor (Classes, Z, Y, X)
        """
        assert image.ndim == 4, f"The input image must be 4D, but got " f"shape {image.shape}."
        Z, Y, X, C = image.shape
        name += f"_z{Z}"
        random_selected_z = np.random.randint(0, Z)
        image = image[random_selected_z].copy()
        image = (image / image.max() * 255).astype(np.uint8)  # (Y, X, C)
        if self.resize is not None:
            image = cv2.resize(image, self.resize, interpolation=cv2.INTER_LINEAR)
        
        if data_sample is not None:
            if "gt_sem_seg" in data_sample:
                assert data_sample.gt_sem_seg.data.shape[-3:] == torch.Size([Z, Y, X])
                gt_sem_seg_2d = data_sample.gt_sem_seg.data[:, random_selected_z].to(torch.uint8)
                if self.resize is not None:
                    gt_sem_seg_2d = F.interpolate(gt_sem_seg_2d[None], self.resize, mode="nearest").squeeze(0)

            if "pred_sem_seg" in data_sample:
                assert data_sample.pred_sem_seg.data.shape[-3:] == torch.Size([Z, Y, X])
                pred_sem_seg_2d = data_sample.pred_sem_seg.data[:, random_selected_z].to(torch.uint8)
                if self.resize is not None:
                    pred_sem_seg_2d = F.interpolate(pred_sem_seg_2d[None], self.resize, mode="nearest").squeeze(0)

            data_sample_2D = SegDataSample(
                gt_sem_seg=PixelData(data=gt_sem_seg_2d),
                pred_sem_seg=PixelData(data=pred_sem_seg_2d),
            )
            data_sample_2D.set_metainfo(data_sample.metainfo)

        else:
            data_sample_2D = None

        return super().add_datasample(name, image, data_sample_2D, *args, **kwargs)


class PackSeg3DInputs(PackSegInputs):
    def transform(self, results: dict) -> dict:
        """Method to pack the input data for 3D segmentation.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`Seg3DDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 4:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(3, 0, 1, 2)))
            else:
                img = img.transpose(3, 0, 1, 2)
                img = to_tensor(img).contiguous()
            packed_results["inputs"] = img

        data_sample = Seg3DDataSample()
        if "gt_seg_map" in results:
            if len(results["gt_seg_map"].shape) == 3:
                data = to_tensor(results["gt_seg_map"][None].astype(np.uint8))
            else:
                warnings.warn(
                    "Please pay attention your ground truth "
                    "segmentation map, usually the segmentation "
                    "map is 3D, but got "
                    f'{results["gt_seg_map"].shape}'
                )
                data = to_tensor(results["gt_seg_map"].astype(np.uint8))
            data_sample.gt_sem_seg = VolumeData(data=data)  # type: ignore
        if "gt_seg_map_one_hot" in results:
            assert len(results["gt_seg_map_one_hot"].shape) == 4, (
                f"The shape of gt_seg_map_one_hot should be `[X, Y, Z, Classes]`, "
                f'but got {results["gt_seg_map_one_hot"].shape}'
            )
            data = to_tensor(results["gt_seg_map_one_hot"].astype(np.uint8))
            data_sample.gt_sem_seg_one_hot = VolumeData(data=data)

        if "seg_fields" in results:
            for key in results['seg_fields']:
                if key not in data_sample.keys():
                    other_seg = BaseDataElement(data=to_tensor(results[key]))
                    data_sample.set_field(other_seg, key)

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results


class Seg3DDataPreProcessor(SegDataPreProcessor):
    """Data preprocessor for 3D segmentation.

    Args:
        mean (Sequence[float]|None): Mean values of input data.
        std (Sequence[float]|None): Standard deviation values of input data.
        size (tuple|None): The size of the input data.
        size_divisor (int|None): The divisor of the size of the input data.
        pad_val (int|float): The padding value of the input data.
        seg_pad_val (int|float): The padding value of the segmentation data.
        batch_augments (list[dict]|None): The batch augmentations for training.
        test_cfg (dict|None): The configuration for testing.
    """

    def __init__(
        self,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        size: tuple | None = None,
        size_divisor: int | None = None,
        pad_val: int | float = 0,
        seg_pad_val: int | float = 255,
        batch_augments: list[dict] | None = None,
        test_cfg: dict | None = None,
        non_blocking: bool = True,
    ):
        super().__init__()
        self._non_blocking = non_blocking
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        if mean is not None:
            assert std is not None, (
                "To enable the normalization in "
                "preprocessing, please specify both "
                "`mean` and `std`."
            )
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std).view(-1, 1, 1, 1), False)
        else:
            self._enable_normalize = False

        self.batch_augments = batch_augments
        self.test_cfg = test_cfg

    @staticmethod
    def stack_batch_3D(
        inputs: list[Tensor],
        data_samples: list[Seg3DDataSample],
        size: tuple | None = None,
        size_divisor: int | None = None,
        pad_val: int | float = 0,
        seg_pad_val: int | float = 255,
    ):
        """Stack multiple 3D volume inputs to form a batch and pad the volumes and gt_sem_segs
        to the max shape using the right bottom padding mode.

        Args:
            inputs (List[Tensor]): The input multiple tensors. each is a
                CZYX 4D-tensor.
            data_samples (list[:obj:`SegDataSample`]): The list of data samples.
                It usually includes information such as `gt_sem_seg`.
            size (tuple, optional): Fixed padding size.
            size_divisor (int, optional): The divisor of padded size.
            pad_val (int, float): The padding value. Defaults to 0
            seg_pad_val (int, float): The padding value. Defaults to 255

        Returns:
        Tensor: The 5D-tensor.
        List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
        """
        assert isinstance(inputs, list), f"Expected input type to be list, but got {type(inputs)}"
        assert len({tensor.ndim for tensor in inputs}) == 1, (
            f"Expected the dimensions of all inputs must be the same, "
            f"but got {[tensor.ndim for tensor in inputs]}")
        assert inputs[0].ndim == 4, f"Expected tensor dimension to be 4, " f"but got {inputs[0].ndim}"
        assert len({tensor.shape[0] for tensor in inputs}) == 1, (
            f"Expected the channels of all inputs must be the same, "
            f"but got {[tensor.shape[0] for tensor in inputs]}")

        padded_inputs = []
        padded_samples = []
        inputs_sizes = [(img.shape[-3], img.shape[-2], img.shape[-1]) for img in inputs]
        max_size = np.stack(inputs_sizes).max(0)
        if size_divisor is not None and size_divisor > 1:
            # the last three dims are Z,Y,X, all subject to divisibility requirement
            max_size = (max_size + (size_divisor - 1)) // size_divisor * size_divisor

        for i in range(len(inputs)):
            tensor = inputs[i]
            if size is not None:
                if len(size) == 2:
                    size = (tensor.shape[-3], *size)
                depth = max(size[-3] - tensor.shape[-3], 0)
                height = max(size[-2] - tensor.shape[-2], 0)
                width = max(size[-1] - tensor.shape[-1], 0)
                # (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
                padding_size = (0, width, 0, height, 0, depth)
            elif size_divisor is not None:
                depth = max(max_size[-3] - tensor.shape[-3], 0)
                height = max(max_size[-2] - tensor.shape[-2], 0)
                width = max(max_size[-1] - tensor.shape[-1], 0)
                padding_size = (0, width, 0, height, 0, depth)
            else:
                padding_size = [0, 0, 0, 0, 0, 0]

            # pad volume
            pad_volume = F.pad(tensor, padding_size, value=pad_val)
            padded_inputs.append(pad_volume)
            # pad gt_sem_seg
            if data_samples is not None:
                data_sample = data_samples[i]
                pad_shape = None
                if "gt_sem_seg" in data_sample:
                    gt_sem_seg = data_sample.gt_sem_seg.data
                    del data_sample.gt_sem_seg.data
                    data_sample.gt_sem_seg.data = F.pad(gt_sem_seg, padding_size, value=seg_pad_val)
                    pad_shape = data_sample.gt_sem_seg.shape
                if "gt_sem_seg_one_hot" in data_sample:
                    gt_sem_seg_one_hot = data_sample.gt_sem_seg_one_hot.data
                    del data_sample.gt_sem_seg_one_hot.data
                    data_sample.gt_sem_seg_one_hot.data = F.pad(gt_sem_seg_one_hot, padding_size, value=0)
                
                data_sample.set_metainfo(
                    {
                        "img_shape": tensor.shape[-3:],
                        "pad_shape": pad_shape,
                        "padding_size": padding_size,
                    }
                )
                padded_samples.append(data_sample)
            
            else:
                padded_samples.append({
                    "img_padding_size": padding_size,
                    "pad_shape": pad_volume.shape[-3:]
                })

        return torch.stack(padded_inputs, dim=0), padded_samples

    def forward(self, data: dict, training: bool = False) -> dict[str, Any]:
        """Perform normalization, padding based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data["inputs"]
        data_samples = data.get("data_samples", None)

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, (
                "During training, ",
                "`data_samples` must be define.",
            )
            inputs, data_samples = self.stack_batch_3D(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val,
            )
            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(inputs, data_samples)
        
        else:
            vol_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == vol_size for input_ in inputs), "The volume size in a batch should be the same."
            
            if self.test_cfg is not None:
                inputs, data_samples = self.stack_batch_3D(
                    inputs=inputs,
                    data_samples=data_samples,
                    size=self.test_cfg.get("size", None),
                    size_divisor=self.test_cfg.get("size_divisor", None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
            else:
                inputs = torch.stack(inputs, dim=0)
        
        return dict(inputs=inputs, data_samples=data_samples)


class PixelUnshuffle1D(torch.nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, inputs: Tensor):
        batch, channels, length = inputs.size()
        r = self.downscale_factor
        out_channels = channels * r
        if length % r != 0:
            raise ValueError(
                f"Input length ({length}) must be divisible by downscale_factor ({r})."
            )
        mid = inputs.view(batch, channels, length // r, r)
        mid = mid.permute(0, 1, 3, 2)
        outputs = mid.contiguous().view(batch, out_channels, length // r)
        return outputs


class PixelShuffle3D(torch.nn.Module):
    def __init__(self, upscale_factor: int|Sequence[int]):
        super(PixelShuffle3D, self).__init__()
        
        if isinstance(upscale_factor, int):
            self.upscale_factor = (upscale_factor, upscale_factor, upscale_factor)
        elif isinstance(upscale_factor, Sequence) and len(upscale_factor) == 3:
            self.upscale_factor = tuple(upscale_factor)
        else:
            raise ValueError(f"upscale_factor必须是一个整数或包含3个整数的序列, 但得到了{upscale_factor}")

    def forward(self, inputs: Tensor):
        # validate
        batch, channels, x, y, z = inputs.size()
        rx, ry, rz = self.upscale_factor
        total_factor = rx * ry * rz
        out_channels = channels // total_factor
        if channels % total_factor != 0:
            raise ValueError(f"输入通道数 ({channels}) 必须能被上采样因子的乘积 ({rx}*{ry}*{rz}={total_factor}) 整除。")
        
        # execute
        mid = inputs.view(batch, out_channels, rx, ry, rz, x, y, z)
        mid = mid.permute(0, 1, 5, 2, 6, 3, 7, 4)
        outputs = mid.contiguous().view(batch, out_channels, x * rx, y * ry, z * rz)
        
        return outputs


class PixelUnshuffle3D(torch.nn.Module):
    def __init__(self, downscale_factor: int|Sequence[int]):
        super(PixelUnshuffle3D, self).__init__()
        
        if isinstance(downscale_factor, int):
            self.downscale_factor = (downscale_factor, downscale_factor, downscale_factor)
        elif isinstance(downscale_factor, Sequence) and len(downscale_factor) == 3:
            self.downscale_factor = tuple(downscale_factor)
        else:
            raise ValueError(f"downscale_factor必须是一个整数或包含3个整数的序列, 但得到了{downscale_factor}")

    def forward(self, inputs: Tensor):
        # validate
        batch, channels, x, y, z = inputs.size()
        rx, ry, rz = self.downscale_factor
        out_channels = channels * (rx * ry * rz)
        if x % rx != 0 or y % ry != 0 or z % rz != 0:
            raise ValueError(f"输入维度 ({x}, {y}, {z}) 必须能被对应的缩放因子 ({rx}, {ry}, {rz}) 整除。")
        
        # execute
        mid = inputs.view(batch, channels, x // rx, rx, y // ry, ry, z // rz, rz)
        mid = mid.permute(0, 1, 3, 5, 7, 2, 4, 6)
        outputs = mid.contiguous().view(batch, out_channels, x // rx, y // ry, z // rz)
        
        return outputs
