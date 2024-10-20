import os.path as osp
import warnings
from numbers import Number
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

import mmcv
from mmcv.transforms import to_tensor
from mmengine.runner import Runner
from mmengine.fileio import get
from mmengine.logging import print_log
from mmengine.structures.base_data_element import BaseDataElement
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses.dice_loss import DiceLoss
from mmseg.datasets.transforms import PackSegInputs
from mmseg.utils import stack_batch



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
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value, (Tensor, np.ndarray)), \
                f'Can not set {type(value)}, only support' \
                f' {(Tensor, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape[-3:]) == self.shape, (
                    'The Z, Y, and X dimensions of '
                    f'values {tuple(value.shape[-3:])} are '
                    'not consistent with '
                    'the shape of this '
                    ':obj:`VolumeData` '
                    f'{self.shape}')
            assert value.ndim in [
                3, 4
            ], f'The dim of value must be 3 or 4, but got {value.ndim}'
            if value.ndim == 3:
                value = value[None]
                warnings.warn('The shape of value will convert from '
                              f'{value.shape[-3:]} to {value.shape}')
            super().__setattr__(name, value)

    def __getitem__(self, item: Sequence[int|slice]) -> 'VolumeData':
        """
        Args:
            item (Sequence[Union[int, slice]]): Get the corresponding values
                according to item.

        Returns:
            :obj:`VolumeData`: Corresponding values.
        """

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):

            assert len(item) == 3, 'Only support to slice Z, Y, and X dimensions'
            tmp_item: list[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(
                        0, slice(single_item, 
                                 None, 
                                 self.shape[-index - 1]) # type: ignore
                        )
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError(
                        'The type of element in input must be int or slice, '
                        f'but got {type(single_item)}')
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError(
                f'Unsupported type {type(item)} for slicing VolumeData')
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
        self.set_field(value, '_gt_sem_seg', dtype=VolumeData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> VolumeData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: VolumeData) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=VolumeData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg

    @property
    def seg_logits(self) -> VolumeData:
        return self._seg_logits

    @seg_logits.setter
    def seg_logits(self, value: VolumeData) -> None:
        self.set_field(value, '_seg_logits', dtype=VolumeData)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._seg_logits



class EncoderDecoder_3D(EncoderDecoder):
    """Encoder Decoder segmentors for 3D data."""

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: list[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If d_crop > d_img or h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxDxHxW,
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

        d_stride, h_stride, w_stride = self.test_cfg.stride # type: ignore
        d_crop, h_crop, w_crop = self.test_cfg.crop_size # type: ignore
        batch_size, _, d_img, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        d_grids = max(d_img - d_crop + d_stride - 1, 0) // d_stride + 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, d_img, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, d_img, h_img, w_img))
        for d_idx in range(d_grids):
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    z1 = d_idx * d_stride
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    z2 = min(z1 + d_crop, d_img)
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    z1 = max(z2 - d_crop, 0)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_vol = inputs[:, :, z1:z2, y1:y2, x1:x2]
                    # change the volume shape to patch shape
                    batch_img_metas[0]['img_shape'] = crop_vol.shape[2:]
                    # the output of encode_decode is seg logits tensor map
                    # with shape [N, C, D, H, W]
                    crop_seg_logit = self.encode_decode(crop_vol, batch_img_metas)
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[4] - x2), int(y1),
                                    int(preds.shape[3] - y2), int(z1), int(preds.shape[2] - z2)))

                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits



class BaseDecodeHead_3D(BaseDecodeHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_seg = torch.nn.Conv3d(
            self.channels, self.out_channels, kernel_size=1)
        if self.dropout_ratio > 0:
            self.dropout = torch.nn.Dropout3d(self.dropout_ratio)



class DiceLoss_3D(DiceLoss):
    def _expand_onehot_labels_dice_3D(
        self, pred: Tensor, target: Tensor) -> Tensor:
        
        """Expand onehot labels to match the size of prediction for 3D Volumes.

        Args:
            pred (torch.Tensor): The prediction, has a shape (N, num_class, D, H, W).
            target (torch.Tensor): The learning label of the prediction,
                has a shape (N, D, H, W).

        Returns:
            torch.Tensor: The target after one-hot encoding,
                has a shape (N, num_class, D, H, W).
        """
        num_classes = pred.shape[1]
        one_hot_target = torch.clamp(target, min=0, max=num_classes)
        one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                    num_classes + 1)
        one_hot_target = one_hot_target[..., :num_classes].permute(0, 4, 1, 2, 3)
        return one_hot_target


    def forward(self, pred, target, *args, **kwargs):
        if (pred.shape != target.shape):
            target = self._expand_onehot_labels_dice_3D(
                pred, target)
        return super().forward(pred, target, *args, **kwargs)



class Seg3DVisualizationHook(SegVisualizationHook):
    def after_val_iter(self, 
                       runner: Runner, 
                       batch_idx: int, 
                       data_batch: dict,
                       outputs: Sequence[Seg3DDataSample]
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
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        window_name = f'val_{osp.basename(img_path)}'

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[Seg3DDataSample]) -> None:
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

            img_path = data_sample.img_path
            window_name = f'test_{osp.basename(img_path)}'

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index)



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
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 4:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(3, 0, 1, 2)))
            else:
                img = img.transpose(3, 0, 1, 2)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = Seg3DDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 3:
                data = to_tensor(results['gt_seg_map'][None].astype(np.uint8))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 3D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.uint8))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = VolumeData(**gt_sem_seg_data)  # type: ignore

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

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
        mean: Sequence[float]|None = None,
        std: Sequence[float]|None = None,
        size: tuple|None = None,
        size_divisor: int|None = None,
        pad_val: int|float = 0,
        seg_pad_val: int|float = 255,
        batch_augments:list[dict]|None = None,
        test_cfg: dict|None = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1, 1), False)
        else:
            self._enable_normalize = False

        self.batch_augments = batch_augments
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> dict[str, Any]:
        """Perform normalization, padding based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments( # type: ignore
                    inputs, data_samples)
        else:
            vol_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == vol_size for input_ in inputs),  \
                'The volume size in a batch should be the same.'
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info}) # type: ignore
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)
