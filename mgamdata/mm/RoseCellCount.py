import pdb
from collections.abc import Sequence

import numpy as np

from mmengine.evaluator.metric import BaseMetric
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.utils import resize



class AccuCount(BaseMetric):
    """
    Designed for counting the number of cells in a given image.
    The counts is calculated by directly add all pixels together.
    """
    def __init__(self, amplify:float=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amplify = amplify

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred = data_sample["seg_logits"]["data"].sum() / self.amplify
            label = data_sample["gt_sem_seg"]["data"].sum()
            self.results.append({"pred_count": pred, "gt_count": label})

    def compute_metrics(self, results: list[dict[str, np.ndarray]]) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
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
        }


class CellCounter(EncoderDecoder):
    def __init__(self, amplify, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amplify = amplify
    
    def postprocess_result(self, seg_logits, data_samples):
        """Delete post-process sigmoid activation when C=1"""
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            i_seg_logits /= self.amplify
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_logits})
            })

        return data_samples


class CellCounterClassifier(CellCounter):
    def __init__(self, amplify, ClasterClassifier):
        ...
