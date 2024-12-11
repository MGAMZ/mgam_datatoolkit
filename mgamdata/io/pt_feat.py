import os
import pickle
import pdb
from typing_extensions import Literal
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
    def __init__(self, field:list[Literal["pt", "csv"]]):
        self.field = field if isinstance(field, list) else [field]
    
    def transform(self, results:dict):
        if "pt" in self.field and "feat_csv" in self.field:
            feat_pt = results["feat_pt"]
            feat_csv = torch.from_numpy(results["feat_csv"])
            results["img"] = torch.cat([feat_csv, feat_pt])
        elif "pt" in self.field:
            results["img"] = results["feat_pt"]
        elif "csv" in self.field:
            results["img"] = torch.from_numpy(results["feat_csv"])
        else:
            raise ValueError(f"No valid field: {self.field}")
        return results
