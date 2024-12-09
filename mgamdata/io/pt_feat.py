import os
import pickle
import pdb
import tqdm

import numpy as np
import torch
from torch import Tensor

from mmcv.transforms import BaseTransform


class LoadPtFeat(BaseTransform):
    def transform(self, results: dict):
        feat_pt_path = results["feat_pt_path"]
        try:
            results["feat_pt"] = torch.load(feat_pt_path, weights_only=False)  # [N, 1024]
        except Exception as e:
            raise RuntimeError(f"Failed to load feature from {feat_pt_path}: {e}")

        return results


class FuseFeat(BaseTransform):
    def transform(self, results:dict):
        feat_pt = results["feat_pt"].mean(dim=0)
        feat_csv = torch.from_numpy(results["feat_csv"])
        results["img"] = torch.cat([feat_csv, feat_pt])
        return results
