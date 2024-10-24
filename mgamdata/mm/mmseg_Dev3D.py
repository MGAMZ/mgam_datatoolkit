import pdb
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any
from tqdm import tqdm

import cv2
import numpy as np
import torch
from torch import Tensor, Type
from torch.nn import functional as F

from mmcv.transforms import to_tensor, Resize, BaseTransform
from mmengine.runner import Runner
from mmengine.structures.base_data_element import BaseDataElement
from mmseg.engine.hooks import SegVisualizationHook
from mmseg.datasets.transforms import PackSegInputs
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses.dice_loss import DiceLoss
from mmseg.models.losses.accuracy import accuracy
from mmseg.visualization.local_visualizer import SegLocalVisualizer
from mmseg.structures.seg_data_sample import SegDataSample, PixelData



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
    def slide_inference(
        self, 
        inputs: Tensor,
        batch_img_metas: list[dict],
        ) -> Tensor:
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

        accu_device:str = self.test_cfg.slide_accumulate_device
        d_stride, h_stride, w_stride = self.test_cfg.stride # type: ignore
        d_crop, h_crop, w_crop = self.test_cfg.crop_size # type: ignore
        batch_size, _, d_img, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        d_grids = max(d_img - d_crop + d_stride - 1, 0) // d_stride + 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = torch.zeros(
            size=(batch_size, out_channels, d_img, h_img, w_img),
            dtype=torch.float16,
            device=accu_device)
        count_mat = torch.zeros(
            size=(batch_size, 1, d_img, h_img, w_img),
            dtype=torch.uint8,
            device=accu_device)
        
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
                    crop_seg_logit = self.encode_decode(
                        crop_vol, batch_img_metas).to(accu_device, non_blocking=True)
                    preds[:, :, z1:z2, y1:y2, x1:x2] += crop_seg_logit
                    count_mat[:, :, z1:z2, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits


    def postprocess_result(self,
                        seg_logits: Tensor,
                        data_samples: list[Seg3DDataSample]|None = None) -> list[Seg3DDataSample]:
        """ Convert results list to `SegDataSample` for 3D Volume segmentation.
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
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 6)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back = padding_size
                # i_seg_logits shape is 1, C, Z, Y, X after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                        padding_front:Z - padding_back,
                                        padding_top:Y - padding_bottom,
                                        padding_left:X - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical', 'depth']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(4, ))
                    elif flip_direction == 'vertical':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = F.interpolate(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='trilinear').squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                            self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                VolumeData(**{'data': i_seg_logits}), # type: ignore
                'pred_sem_seg':
                VolumeData(**{'data': i_seg_pred}) # type: ignore
            })

        return data_samples



class BaseDecodeHead_3D(BaseDecodeHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_seg = torch.nn.Conv3d(
            self.channels, self.out_channels, kernel_size=1)
        if self.dropout_ratio > 0:
            self.dropout = torch.nn.Dropout3d(self.dropout_ratio)    

    @abstractmethod
    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor]:
        ...


    def loss_per_layer(self,
                       seg_logit:Tensor,
                       seg_label:Tensor,
                       losses_dict:dict,
                       weight:float=1.,
                      ) -> dict:
        # seg_logit = F.interpolate(
        #     input=seg_logit,
        #     size=seg_label.shape[2:],
        #     mode='trilinear')
        seg_label = F.interpolate(
            input=seg_label,
            size=seg_logit.shape[2:],
            mode='nearest')
        
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
                losses_dict[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index) * weight
            else:
                losses_dict[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index) * weight

        return losses_dict


    def loss(self,
             inputs: tuple[Tensor],
             batch_data_samples: list[Seg3DDataSample],
             train_cfg: dict) -> dict:
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
        # list of Tensor: [B, C, Z, Y, X]
        seg_logits = self.forward(inputs)
        # [B, C, Z, Y, X]
        seg_label = self._stack_batch_gt(batch_data_samples) # type: ignore
        
        losses = dict()
        
        for i, seg_logit in enumerate(seg_logits):
            losses = self.loss_per_layer(
                seg_logit, seg_label, losses, weight=1/(2**i))
        
        losses['acc_seg'] = accuracy(
            seg_logits[0],
            seg_label.squeeze(1),
            ignore_index=self.ignore_index)

        return losses


    def predict(self,
                inputs: tuple[Tensor],
                batch_img_metas: list[dict],
                test_cfg: dict) -> Tensor:
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
        seg_logits = self.forward(inputs)[0]    # Select the last output, shape: [B, C, Z, Y, X]

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = F.interpolate(
            input=seg_logits,
            size=size,
            mode='trilinear',
            align_corners=self.align_corners)
        return seg_logits



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
        one_hot_target = torch.nn.functional.one_hot(
            one_hot_target.to(torch.int64), num_classes+1)
        one_hot_target = one_hot_target[..., :num_classes].permute(0, 4, 1, 2, 3)
        return one_hot_target


    def forward(self, pred, target, *args, **kwargs):
        if (pred.shape != target.shape):
            target = self._expand_onehot_labels_dice_3D(
                pred, target)
            assert pred.shape == target.shape
        return super().forward(pred, target, *args, **kwargs)



class Seg3DVisualizationHook(SegVisualizationHook):
    def after_val_iter(self, 
                       runner: Runner, 
                       batch_idx: int, 
                       data_batch: dict,
                       outputs: Sequence[Seg3DDataSample]
                       ) -> None:
        if self.draw is False:
            return
        total_curr_iter = runner.iter + batch_idx

        # NOTE Override original implementation.
        # data batch inputs [N, C, Z, Y, X], but requires RGB at last dimension.
        img = data_batch['inputs'][0].permute(1,2,3,0).numpy()
        # img: [Z, Y, X, 1] -> [Z, Y, X, 3]
        img = np.repeat(img, 3, axis=-1)
        series_id = outputs[0].metainfo['series_id']
        window_name = f'val_{series_id}'
        
        if (total_curr_iter % self.interval == 0) or (total_curr_iter == 1):
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
            
            # NOTE Override original implementation.
            # data batch inputs [N, C, Z, Y, X], but requires RGB at last dimension.
            img = data_batch['inputs'][0].permute(1,2,3,0).numpy()
            # img: [Z, Y, X, 1] -> [Z, Y, X, 3]
            img = np.repeat(img, 3, axis=-1)
            series_id = outputs[0].metainfo['series_id']
            window_name = f'val_{series_id}'

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index)



class Seg3DLocalVisualizer(SegLocalVisualizer):
    def __init__(self,
                 name,
                 resize:Sequence[int]|None=None,
                 *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.resize = resize
    
    
    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Seg3DDataSample|None=None,
        *args, **kwargs
    ) -> None:
        """ Randomly select a slice from the 3D volume and fall back to 2D visualize.
        
        Args:
            name: ...
            image (np.ndarray): The image to visualize, NdArray (Z, Y, X, C).
            data_sample (Seg3DDataSample, optional): The data sample to visualize.
                - gt_sem_seg (data:VolumeData): tensor (1, Z, Y, X)
                - pred_sem_seg (data:VolumeData): tensor (1, Z, Y, X)
                - seg_logits (data:VolumeData): tensor (Classes, Z, Y, X)
        """
        assert image.ndim == 4, f'The input image must be 4D, but got '\
                                f'shape {image.shape}.'
        Z, Y, X, C = image.shape
        name += f'_z{Z}'
        random_selected_z = np.random.randint(0, Z)
        image = np.take(image, random_selected_z, axis=0)
        image = (image/image.max()*255).astype(np.uint8) # (Y, X, C)
        if self.resize is not None:
            image = cv2.resize(image, self.resize, interpolation=cv2.INTER_LINEAR)
        
        if data_sample is not None:
            if 'gt_sem_seg' in data_sample:
                assert data_sample.gt_sem_seg.data.shape[-3:] == torch.Size([Z, Y, X])
                gt_sem_seg_2d = data_sample.gt_sem_seg.data[:, random_selected_z].to(torch.uint8)
                if self.resize is not None:
                    gt_sem_seg_2d = F.interpolate(
                        gt_sem_seg_2d[None], self.resize, mode='nearest').squeeze(0)
            
            if 'pred_sem_seg' in data_sample:
                assert data_sample.pred_sem_seg.data.shape[-3:] == torch.Size([Z, Y, X])
                pred_sem_seg_2d = data_sample.pred_sem_seg.data[:, random_selected_z].to(torch.uint8)
                if self.resize is not None:
                    pred_sem_seg_2d = F.interpolate(
                        pred_sem_seg_2d[None], self.resize, mode='nearest').squeeze(0)
            
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

    @staticmethod
    def stack_batch_3D(
        inputs: list[Tensor],
        data_samples: list[Seg3DDataSample]|None = None,
        size: tuple|None = None,
        size_divisor: int|None = None,
        pad_val: int|float = 0,
        seg_pad_val: int|float = 255):
        
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
        assert isinstance(inputs, list), \
            f'Expected input type to be list, but got {type(inputs)}'
        assert len({tensor.ndim for tensor in inputs}) == 1, \
            f'Expected the dimensions of all inputs must be the same, ' \
            f'but got {[tensor.ndim for tensor in inputs]}'
        assert inputs[0].ndim == 4, f'Expected tensor dimension to be 4, ' \
            f'but got {inputs[0].ndim}'
        assert len({tensor.shape[0] for tensor in inputs}) == 1, \
            f'Expected the channels of all inputs must be the same, ' \
            f'but got {[tensor.shape[0] for tensor in inputs]}'

        # only one of size and size_divisor should be valid
        assert (size is not None) ^ (size_divisor is not None), \
            'only one of size and size_divisor should be valid'

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
                if 'gt_sem_seg' in data_sample:
                    gt_sem_seg = data_sample.gt_sem_seg.data
                    del data_sample.gt_sem_seg.data
                    data_sample.gt_sem_seg.data = F.pad(
                        gt_sem_seg, padding_size, value=seg_pad_val)
                    pad_shape = data_sample.gt_sem_seg.shape
                data_sample.set_metainfo({
                    'img_shape': tensor.shape[-3:],
                    'pad_shape': pad_shape,
                    'padding_size': padding_size
                })
                padded_samples.append(data_sample)
            else:
                padded_samples.append(
                    dict(
                        img_padding_size=padding_size,
                        pad_shape=pad_volume.shape[-3:]))

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
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = self.stack_batch_3D(
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
                inputs, padded_samples = self.stack_batch_3D(
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



class Resize3D(Resize):
    @staticmethod
    def scale_2D_or_3D(original_shape:list[int], target_shape:list[int]):
        if len(original_shape) == len(target_shape) + 1:
            return [original_shape[0], *target_shape]
        elif len(original_shape) == len(target_shape):
            return target_shape
        else:
            raise ValueError(
                'The dimension of the segmentation map should be equal '
                'to the scale dimension or the scale dimension plus 1, '
                f'but got {original_shape} and {target_shape}')


    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        for seg_key in results.get('seg_fields', []):
            if results.get(seg_key, None) is not None:
                scale = self.scale_2D_or_3D(results[seg_key].shape, results['scale'])
                original = torch.from_numpy(results[seg_key])
                results[seg_key] = F.interpolate(
                    original[None, None],
                    size=scale,
                    mode='nearest'
                )[0,0].numpy()


    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        if results.get('img', None) is not None:
            scale = self.scale_2D_or_3D(results['img'].shape, results['scale'])
            original = torch.from_numpy(results['img'].astype(np.float32))
            img = F.interpolate(
                original[None, None],
                size=scale,
                mode='trilinear')
            
            results['img'] = img[0,0].numpy().astype(results['img'].dtype)
            results['img_shape'] = img.shape
            results['scale_factor'] = [
                new / ori for new, ori in zip(
                    results['img_shape'], results['ori_shape'])]



class RandomCrop3D(BaseTransform):
    """Random crop the 3D volume & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int, int]]):  Expected size after cropping
            with the format of (d, h, w). If set to an integer, then cropping
            depth, width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """
    def __init__(self,
                 crop_size: int|tuple[int, int, int],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        if isinstance(crop_size, Sequence):
            assert len(crop_size) == 3, \
                f'The expected crop_size containing 3 integers, but got {crop_size}'
        elif isinstance(crop_size, int):
            crop_size = (crop_size, crop_size, crop_size)
        else:
            raise TypeError(f"Unsupported crop size: {crop_size}")
        
        assert min(crop_size) > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index


    def crop_bbox(self, results: dict, failed_times:int=0) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped volume.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input volume.

            Returns:
                tuple: Coordinates of the cropped volume.
            """

            margin_d = max(img.shape[0] - self.crop_size[0], 0)
            margin_h = max(img.shape[1] - self.crop_size[1], 0)
            margin_w = max(img.shape[2] - self.crop_size[2], 0)
            offset_d = np.random.randint(0, margin_d + 1)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_d1, crop_d2 = offset_d, offset_d + self.crop_size[0]
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[1]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[2]

            return crop_d1, crop_d2, crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for crop_time in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if (len(cnt) > 1) and \
                    ((np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio):
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox


    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input volume.
            crop_bbox (tuple): Coordinates of the cropped volume.

        Returns:
            np.ndarray: The cropped volume.
        """

        crop_d1, crop_d2, crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_d1:crop_d2, crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img


    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop volumes, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the volume
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:3]
        return results


    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'



