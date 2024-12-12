import os
import pdb
from typing_extensions import Literal

import torch

from mmcv.transforms import BaseTransform


class FuseFeat(BaseTransform):
    def __init__(self, field:list[Literal["pt", "csv"]]):
        self.field = field if isinstance(field, list) else [field]
    
    def transform(self, results:dict):
        feat_pt_path = results["feat_pt_path"]
        if feat_pt_path is not None and "pt" in self.field:
            try:
                results["img"] = torch.load(feat_pt_path, weights_only=False)  # [N, 1024]
            except Exception as e:
                raise RuntimeError(f"Failed to load feature from {feat_pt_path}: {e}")
        
        if "csv" in self.field:
            results["feat_csv"] = torch.from_numpy(results["feat_csv"])
        
        return results
