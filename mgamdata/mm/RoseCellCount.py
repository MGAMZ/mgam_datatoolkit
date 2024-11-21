from collections.abc import Sequence

import numpy as np

from mmengine.evaluator.metric import BaseMetric


class AccuCount(BaseMetric):
    """
    Designed for counting the number of cells in a given image.
    The counts is calculated by directly add all pixels together.
    """
    def __init__(self, amplify, *args, **kwargs):
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
            pred_label = data_sample["seg_logits"]["data"].sum() / self.amplify
            label = data_sample["gt_sem_seg"]["data"].sum()
            self.results.append({"pred_count": pred_label, "gt_count": label})

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
