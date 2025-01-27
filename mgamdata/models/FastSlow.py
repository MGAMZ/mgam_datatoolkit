"""
Author: Yiqin Zhang
Initiate Time: 2024.11
Email: 312065559@qq.com

This is the implementation of our research.
"""


import os
import pdb
import math
import torch.utils.checkpoint
from typing_extensions import Literal
from itertools import permutations

import numpy as np
import torch
import mpl_toolkits.mplot3d.art3d as art3d
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.axes import Axes
from matplotlib.gridspec import SubplotSpec

from mmcv.transforms import BaseTransform
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.dist import master_only
from mmengine.visualization import Visualizer
from mmengine.evaluator import BaseMetric
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample

from .SelfSup import AutoEncoderSelfSup
from ..mm.mmeng_PlugIn import MomentumAvgModel, GeneralViser


DIM_MAP = {"1d": 1, "2d": 2, "3d": 3}
CMAP_SEQ_COLOR = ["winter", "RdPu"]
DEFAULT_CMAP = plt.get_cmap(CMAP_SEQ_COLOR[0])
CMAP_COLOR = [DEFAULT_CMAP(32), DEFAULT_CMAP(224)]
CONV_MAPPING = {
    "3d": nn.Conv3d,
    "2d": nn.Conv2d,
    "1d": nn.Conv1d
}



class RandomVolumeView(BaseTransform):
    """
        Get random view from the original image.
        
        Required fields:
            - img: [C, *]
            - seg: [*] (Optional)
            - seg_fields
        
        Modified fields:
            - img: [num_views, C, *]
            - seg: [num_views, *] (Optional)
        
        Added fields:
            - view_coords: [num_views, num_spatial_dims]
    """
    def __init__(self, 
                 num_views: int, 
                 dim: Literal["1d", "2d", "3d"], 
                 size: tuple[int]):
        self.num_views = num_views
        self.dim = dim
        self.size = size
    
    def _determine_slices(self, volume_shape: tuple[int]) -> tuple[list[slice], list[int]]:
        full_slices = [slice(None)] * len(volume_shape)
        center_coords = []
        
        for i, s in enumerate(self.size):
            dim_idx = -(i + 1)
            start = np.random.randint(0, volume_shape[dim_idx] - s + 1)
            full_slices[dim_idx] = slice(start, start + s)
            center_coords.insert(0, start + s // 2)
        
        return full_slices, center_coords
    
    def _get_view(self, array: np.ndarray, slices: list[slice]) -> np.ndarray:
        return array[tuple(slices)]
    
    def transform(self, results):
        img = results["volume"] = results["img"]
        segs = {seg_field: [] for seg_field in results.get("seg_fields", [])}
        coords = []
        img_views = []
        
        for _ in range(self.num_views):
            slices, center_coord = self._determine_slices(img.shape)
            
            img_view = self._get_view(img, slices)
            img_views.append(img_view)
            coords.append(center_coord)
            
            for seg_field in results.get("seg_fields", []):
                seg_slices = slices[1:] if len(slices) > 1 else slices
                sub_seg = self._get_view(results[seg_field], seg_slices)
                segs[seg_field].append(sub_seg)
        
        results["img"] = np.stack(img_views)  # [num_views, *]
        # [num_views, num_spatial_dims]
        results["view_coords"] = np.array(coords).astype(np.int32)
        for seg_field in results.get("seg_fields", []):
            results[seg_field] = np.stack(segs[seg_field])
        
        return results


class NormalizeCoord(BaseTransform):
    """
    Normalize the coordination.
    
    Required fields:
        - view_coords: [num_views, num_spatial_dims]
    
    Modified fields:
        - view_coords: [num_views, num_spatial_dims]
    """
    def __init__(self, div: list[int]):
        self.div = div
    
    def _normalize(self, coords: np.ndarray) -> np.ndarray:
        for i, s in enumerate(self.div):
            coords[:, i] = coords[:, i] / s
        return coords
    
    def transform(self, results:dict):
        results["normed_view_coords"] = self._normalize(
            results["view_coords"].copy().astype(np.float32))
        return results


class ParseCoords(BaseTransform):
    """
    Pre-Parse the generated coordination context,
    to minimize train-time label generation.
    """
    def __init__(self, view_size: tuple[int], sub_view_size: tuple[int]):
        self.view_size = np.array(view_size)
        self.sub_view_size = np.array(sub_view_size)
    
    def _get_AdjDst_indices(self, abs_gap:np.ndarray, coords:np.ndarray) -> tuple:
        """
        Locate the adjacent pair and distant pair of two sub-view.
        The results contain view indexs and volume absolute coords.
        
        Args:
            abs_gap (Tensor): [N, num_views, num_views, coord-dims]
            coords (Tensor): [N, num_views, coord-dims]
            volume_shape (list[int]): [N, ...]
        Returns:
            tuple[list[list[tuple]], Tensor]: 
                - indices: [N, num_pairs, 4, index (tuple) (view_idx, *coord-dim-slices)]
                - centers: [N, num_pairs, 4, coord-dims]
        """
        # prepare for the calculation
        num_views, cdims = coords.shape
        pairs = [(i, j) for i in range(num_views) for j in range(i+1, num_views)]
        num_pairs = len(pairs)
        indices = [None] * num_pairs
        centers = np.zeros((num_pairs, 4, cdims))
        
        def sub_area_center(direction:Tensor) -> Tensor:
            """
            Given the sub-view size and the direction vector.
            It is possible to determine the maximum center offset along the direction,
            before the sub-view touches the view's boundary.
            
            Args:
                direction (Tensor): [..., coord-dims]
            Return:
                offset (Tensor): [..., coord-dims]
            """
            # The absolute gap available for the center to move 
            # before touching the boundary on each dimension.
            avail_gap = (self.view_size - self.sub_view_size) / 2
            # Calculate the relative gap corresponding to direction vector
            rel_direction_ratio = avail_gap / (direction + 1e-5)
            # Determine the maximum relative ratio.
            # The minimum value of all dimensions are used,
            # because it will be the first dimension to hit the boundary.
            max_direction_ratio = np.abs(rel_direction_ratio).min(axis=0)
            # Return the absolute maximum offset on each dimension.
            return max_direction_ratio * direction

        offsets = np.round(sub_area_center(abs_gap)).astype(np.int32)

        # Traverse all possible view pairs among all views.
        for idx_pair, (i, j) in enumerate(pairs):
            # Get the maximum offset of the pair.
            # offset = torch.round(sub_area_center(abs_gap[i, j])).int()
            offset = offsets[i, j]
            
            # Calculate the center of the adjacent pair and the distant pair.
            # Volume coordinate system here.
            centers[idx_pair, 0] = coords[i] + offset  # adj1
            centers[idx_pair, 1] = coords[j] - offset  # adj2
            centers[idx_pair, 2] = coords[i] - offset  # dst1
            centers[idx_pair, 3] = coords[j] + offset  # dst2
            
            # Calculate the sub-view index of the adjacent pair and the distant pair.
            # View coordinate system here. (namely patch)
            v_c = self.view_size//2  # view center
            sub_v_c = self.sub_view_size//2  # sub-view center
            # v_c +/- derterimines the sub-view center
            # and the center +/- sub_v_c determines the sub-view boundary,
            # then slice index is available to directly sample from view array.
            indices[idx_pair] = [
                # adj1
                (i, ) + tuple(slice(v_c[x] + offset[x] - sub_v_c[x], 
                                    v_c[x] + offset[x] + sub_v_c[x]
                        ) for x in range(cdims)),
                # adj2
                (j, ) + tuple(slice(v_c[x] - offset[x] - sub_v_c[x], 
                                    v_c[x] - offset[x] + sub_v_c[x]
                        ) for x in range(cdims)),
                # dst1
                (i, ) + tuple(slice(v_c[x] - offset[x] - sub_v_c[x], 
                                    v_c[x] - offset[x] + sub_v_c[x]
                        ) for x in range(cdims)),
                # dst2
                (j, ) + tuple(slice(v_c[x] + offset[x] - sub_v_c[x], 
                                    v_c[x] + offset[x] + sub_v_c[x]
                        ) for x in range(cdims)),
            ]

        return indices, centers

    def _parse_gap(self, coords:np.ndarray):
        # coords (coordinates):        [3 (sub-volume), 3 (coord-dim)]
        # abs_gap (absolute distance): [3 (start from), 3 (point to), 3(coord-dim)]
        abs_gap = coords[None] - coords[:, None]
        # rel_gap (relative distance): [3 (start from), 3 (point to), 3(coord-dim)]
        # The gap matrix is symmetric, so we can use the upper triangle part.
        # The following implementation is a trick, which will get relative value 
        # when comparing with the max gap value on each dimension.
        rel_base = abs_gap.max(axis=(0,1))  # determine the max gap for each dimension
        rel_gap = abs_gap / (rel_base[None, None, ...] + 1e-5)  # normalize the gap matrix
        
        return coords, abs_gap.astype(np.float32), rel_gap.astype(np.float32)

    def _find_all_paths(self, direction: np.ndarray) -> np.ndarray:
        """
        Args:
            direction (np.ndarray): [L, L, D]
                L: number of points
                D: number of dimensions
        Returns:
            np.ndarray: [L, L, num_paths, D]
        """
        
        L, L, D = direction.shape
        
        # 存储格式: {(start, end): [path_vectors]}
        paths_dict = {(i,j):[] for i in range(L) for j in range(L) if i != j}
        
        def backtrack(current: int, start: int, visited: set, path_vector: np.ndarray):
            if len(visited) == L:
                paths_dict[(start, current)].append(path_vector)
                return
            for next_point in range(L):
                if next_point not in visited:
                    visited.add(next_point)
                    new_vector = path_vector + direction[current, next_point]
                    backtrack(next_point, start, visited, new_vector)
                    visited.remove(next_point)
        
        for start_point in range(L):
            backtrack(start_point, start_point, {start_point}, np.zeros(D))
        
        # 整理数据为张量形式 [L, L, max_paths, D]
        batch_paths = []
        for i in range(L):
            end_paths = []
            for j in range(L):
                if i != j:
                    paths = paths_dict[(i,j)]
                    if paths:
                        end_paths.append(np.stack(paths))
                    else:
                        end_paths.append(np.zeros((1, D)))
                else:
                    # 对角线位置(起点=终点)填充零向量
                    end_paths.append(np.zeros((1, D)))
            
            # 确保每个终点的路径数量一致
            max_paths = max(p.shape[0] for p in end_paths)
            padded_end_paths = []
            for paths in end_paths:
                if paths.shape[0] < max_paths:
                    padding = np.zeros((max_paths - paths.shape[0], D))
                    padded_end_paths.append(np.concatenate([paths, padding], axis=0))
                else:
                    padded_end_paths.append(paths)
            batch_paths.append(np.stack(padded_end_paths))
        
        return np.stack(batch_paths)  # [L, L, num_paths, D]

    def _parse_route(self, direction:np.ndarray) -> np.ndarray:
        
        def _calc_destination(routes:np.ndarray, 
                              coords:np.ndarray, 
                              gapBetweenCoords:np.ndarray):
            """
            Determine the route's destination point coord along each specificed
            mid-point in the routes.
            
            Args:
                routes (np.ndarray): [routes, steps, coord-dims]
                coords (np.ndarray): [points, coord-dims]
                gapBetweenCoords (np.ndarray): [points (from), points (to), coord-dims]
            
            Returns:
                routes' destinations (np.ndarray): [routes, coord-dims]
            """
            num_paths = routes.shape[0]
            coord_dims = coords.shape[1]
            destinations = np.zeros((num_paths, coord_dims), dtype=coords.dtype)
            
            for i in range(num_paths):
                path = routes[i]
                position = coords[path[0, 0]]  # 初始化为path第一个step的start坐标
                for step in path:
                    s, t = step
                    position += gapBetweenCoords[s, t]
                destinations[i] = position
            
            return destinations  # [routes, coord-dims]

    def transform(self, results:dict):
        # [num_views, coord-dims]
        coords = results["view_coords"]
        normed_coords = results["normed_view_coords"]
        
        # gap: [3 (start from), 3 (point to), 3(coord-dim)]
        coords, abs_gap, rel_gap = self._parse_gap(coords)
        results["abs_gap"] = torch.from_numpy(abs_gap)
        results["rel_gap"] = torch.from_numpy(rel_gap)
        
        normed_coords, normed_abs_gap, normed_rel_gap = self._parse_gap(normed_coords)
        results["normed_abs_gap"] = torch.from_numpy(normed_abs_gap)
        results["normed_rel_gap"] = torch.from_numpy(normed_rel_gap)
        
        # indices: [num_pairs, 4, index (tuple) (view_idx, *coord-dim-slices)]
        # centers: [num_pairs, 4, coord-dims]
        results["sim_pair_indices"], results["sim_pair_centers"] = self._get_AdjDst_indices(abs_gap, coords)
        results["normed_sim_pair_indices"], results["normed_sim_pair_centers"] = self._get_AdjDst_indices(normed_abs_gap, normed_coords)
        
        return results


class RelativeSimilaritySelfSup(AutoEncoderSelfSup):
    """The top design of the Self-Supervision."""
    
    def __init__(
        self, 
        gap_head:ConfigDict, 
        sim_head:ConfigDict, 
        vec_head:ConfigDict, 
        momentum=1e-4, 
        gamma=100, 
        update_interval=1, 
        checkpoint_nir:bool=False,
        extra_keys:list[str]=[
            "view_coords", "abs_gap", "rel_gap", "sim_pair_indices", 
            "sim_pair_centers", "normed_view_coords", "normed_abs_gap", "normed_rel_gap", 
            "normed_sim_pair_indices", "normed_sim_pair_centers"],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.extra_keys = extra_keys
        self.gap_head:GapPredictor          = MODELS.build(gap_head)
        self.sim_head:SimPairDiscriminator  = MODELS.build(sim_head)
        self.vec_head:VecAngConstraint      = MODELS.build(vec_head)
        self.checkpoint_nir = checkpoint_nir
        self.momentum = momentum
        if momentum is not None:
            self.momentum_encoder = MomentumAvgModel(
                self.whole_model_,
                momentum=momentum,
                gamma=gamma,
                interval=update_interval,
                update_buffers=False,
            )

    def _stack_coord_info(self, data_samples:list[DataSample]):
        s = {}
        for k in self.extra_keys:
            if isinstance(data_samples[0].get(k), torch.Tensor):
                s[k] = torch.stack([sample.get(k) for sample in data_samples])
            else:
                s[k] = np.stack([sample.get(k) for sample in data_samples])
        return s

    def extract_nir(self, vv_main:Tensor, vv_aux:Tensor) -> Tensor:
        """
        Args:
            nir_main (Tensor): [N, C, Z, Y, X]
            nir_aux (Tensor): [N, view, C, Z, Y, X]
        
        Returns:
            nir (Tensor): [N, view, C, Z, Y, X]
        """
        # vv: volume view
        nir_main = self.whole_model_(vv_main)[0]
        # nir_vv: neural implicit representation of volume view
        aux_model = self.momentum_encoder \
                    if self.momentum \
                    else self.whole_model_
        with torch.no_grad():
            nir_aux = [aux_model(view)[0] 
                       for view in vv_aux.transpose(0, 1)]
        nir = torch.stack([nir_main, *nir_aux], dim=1)  # [N, view, C, ...]
        return nir  # [N, view, C, ...]

    def loss(
        self, inputs:Tensor, data_samples:list[DataSample], **kwargs
    ) -> dict[str, Tensor]:
        """
        Args:
            inputs (Tensor): [N, sub-view, C, *]
            data_samples (list[DataSample]): 
                DataSample:
                    - view_coords (Tensor): [sub-view, 3]
        """
        self.backbone: BaseModule
        
        # vv: volume view
        vv_main = inputs[:, 0]
        vv_aux = inputs[:, 1:]
        coord_info = self._stack_coord_info(data_samples)
        
        # neural implicit representation forward
        if self.checkpoint_nir:
            # [N, sub-view, C, ...]
            nir = torch.utils.checkpoint.checkpoint(
                self.extract_nir, vv_main, vv_aux,
                use_reentrant=False,
            )
        else:
            # [N, sub-view, C, ...]
            nir = self.extract_nir(vv_main, vv_aux)
        
        losses = {}
        # relative gap self-supervision
        gap_losses = self.gap_head.loss(nir, coord_info["normed_abs_gap"])
        for k, v in gap_losses.items():
            losses[k] = v
        # similarity self-supervision (Opposite nir has larger difference, namely negative label)
        sim_losses = self.sim_head.loss(nir, coord_info["sim_pair_indices"])
        for k, v in sim_losses.items():
            losses[k] = v
        # vector self-supervision
        vec_losses = self.vec_head.loss(nir, coord_info["normed_abs_gap"])
        for k, v in vec_losses.items():
            losses[k] = v
        
        # update momentum model
        if self.momentum is not None:
            self.momentum_encoder.update_parameters(self.whole_model_)
        
        return losses

    @torch.inference_mode()
    def predict(self, inputs: Tensor, data_samples: list[DataSample], **kwargs
    ) -> list[DataSample]:
        """
        Args:
            inputs (Tensor): [N, sub-view, C, *]
            data_samples (list[DataSample]): 
                DataSample:
                    - view_coords (Tensor): [sub-view, coord-dims]
                    - volume (np.ndarray): [C, ...]
        
        Returns:
            list[DataSample]:
                DataSample:
                    - volume (np.ndarray): [view, C, ...]
                    - rel_gap (Tensor): [view (start from), view (point to), coord-dims]
                    - abs_gap (Tensor): [view (start from), view (point to), coord-dims]
                    - coords_gt (Tensor): [view, coord-dims]
                    - view_coords (Tensor): [view, coord-dims]
                    - gap_pred (Tensor): [view, view]
                    - sim_pred (Tensor): [num_pairs, 4 (adj1, adj2, dst1, dst2), coord-dims]
                    - vec_pred (Tensor): [view, view, coord-dims]
        """
        coord_info = self._stack_coord_info(data_samples)

        vv_main = inputs[:, 0]
        vv_aux = inputs[:, 1:]
        # neural implicit representation forward
        nir = self.extract_nir(vv_main, vv_aux)  # [N, view, C, ...]

        # [N, view, view]
        gap_pred, gap_loss = self.gap_head.predict(nir, coord_info["normed_abs_gap"])
        # [N, num_pairs, 4 (i_adj1, i_adj2, i_dist1, i_dist2)]
        sim_pred, sim_loss = self.sim_head.predict(nir, coord_info["sim_pair_indices"], coord_info["sim_pair_centers"])
        # [N, view, view, 3]
        vec_pred, vec_loss = self.vec_head.predict(nir, coord_info["normed_abs_gap"])
        
        for i in range(len(data_samples)):
            data_samples[i].sub_views = inputs[i]
            data_samples[i].nir = nir[i]
            data_samples[i].gap_pred = gap_pred[i]
            data_samples[i].sim_pred = sim_pred[i]
            data_samples[i].vec_pred = vec_pred[i]
            data_samples[i].sim_chunk_size = self.sim_head.sub_view_size
            data_samples[i].gap_loss = gap_loss
            data_samples[i].sim_loss = sim_loss
            data_samples[i].vec_loss = vec_loss

        return data_samples
    
    def forward(self,
                inputs: Tensor|list[Tensor],
                data_samples: list[DataSample],
                mode: str = 'tensor'):
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')


class GlobalAvgPool(nn.Module):    
    def __init__(self, dim: Literal["1d", "2d", "3d"]):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == "1d":
            return nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        elif self.dim == "2d":
            return nn.functional.adaptive_avg_pool2d(x, (1,1)).squeeze(-1).squeeze(-1)
        elif self.dim == "3d":
            return nn.functional.adaptive_avg_pool3d(x, (1,1,1)).squeeze(-1).squeeze(-1).squeeze(-1)
        else:
            raise NotImplementedError(f"Invalid Dim Setting: {self.dim}")


class BaseVolumeWisePredictor(nn.Module):
    """
    The class is shared by `GapPredictor` and `VecAngConstraint`.
    They both need feature extraction for views.
    """
    
    def __init__(self, dim:Literal["1d","2d","3d"], in_channels:int, num_views:int=3):
        super().__init__()
        
        self.dim = dim
        self.num_views = num_views
        self.act = nn.GELU()
        self.avg_pool = GlobalAvgPool(dim)
        
        # 预计算通道数
        self.channels = []
        c = in_channels
        for _ in range(4):
            self.channels.append(c)
            c = c // 2
        
        for i in range(3):
            setattr(
                self, f"proj_{i+1}", 
                CONV_MAPPING[dim](
                    in_channels=self.channels[i],
                    out_channels=self.channels[i+1],
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
    
    def forward(self, x:Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, num_views, C, Z, Y, X]
        
        Returns:
            x (Tensor): [N, num_views, C]
        """
        views = list(x.unbind(dim=1))
        for i, view in enumerate(views):
            for j in range(3):
                view:_ConvNd = getattr(self, f"proj_{j+1}")(view)
                view = self.act(view)
            view = self.avg_pool(view)

            views[i] = view
        
        return torch.stack(views, dim=1)  # [N, num_views, C]


class GapPredictor(BaseVolumeWisePredictor):
    """Predict the gap between all views."""
    
    def __init__(self, loss_weight:float=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cri = nn.MSELoss()
        self.loss_weight = loss_weight
        self.proj_axis = nn.Linear(self.channels[-1], 3)
    
    def forward(self, nir:Tensor) -> Tensor:
        """
        Args:
            nir (Tensor): Size [N, num_views, C, Z, Y, X]
        
        Returns:
            axis_gap (Tensor): Size [N, num_views, num_views, 3]
        """
        nir = super().forward(nir)  # [N, num_views, C]
        
        # The origin may align with the mean value of all samples' world coordinate systems'origin.
        # diff equals to the relative distance between each `nir`.
        rep_diff = nir.unsqueeze(2) - nir.unsqueeze(1)  # (N, num_views, num_views, C)
        # calculate the distance of `rel_pos_rep_diff`
        axis_gap = self.proj_axis(rep_diff)  # (N, num_views, num_views, coord-dim-length)
        return axis_gap  # (N, num_views, num_views)
    
    def loss(self, nir:Tensor, gap:Tensor) -> dict[str, Tensor]:
        """
        Args:
            nir (Tensor): Size [N, num_views, C, ...]
            gap (Tensor): Size [N, num_views (start from), num_views (point to), coord-dim-length]
        """
        # (N, num_views, num_views, coord-dim-length)
        axis_gap = self.forward(nir)
        # (N, num_views, num_views, coord-dim-length)
        loss_axis_gap = self.cri(axis_gap, gap)
        loss_norm_gap = self.cri(axis_gap.norm(dim=-1), gap.norm(dim=-1))
        return {"loss/axis_gap": loss_axis_gap*self.loss_weight,
                "loss/norm_gap": loss_norm_gap*self.loss_weight}

    @torch.inference_mode()
    def predict(self, nir:Tensor, gap:Tensor|None=None) -> tuple[Tensor, Tensor|None]:
        """
        Args:
            nir (Tensor): [N, num_views, C, ...]
            gap (Tensor): Size [N, num_views (start from), num_views (point to), coord-dim-length]
            
        Returns:
            gap_pred (Tensor): [N, num_views, num_views]
            gap_loss (Tensor): [1, ]
        """
        pred = self.forward(nir).norm(dim=-1)
        if gap is not None:
            loss = self.cri(pred, gap.norm(dim=-1))
        else:
            loss = None
        return pred, loss


class SimPairDiscriminator(BaseModule):
    """
    Predict the similarity between adjacent and distant pairs.
    The pairs are generated among all combinations of the sub-views.
    A pair of sub-views can generate an adjacent and a distant pair.
    """
    
    LABEL_ADJA_PAIR = 0
    LABEL_DIST_PAIR = 1
    
    def __init__(self, 
                 view_size:int|list[int], 
                 sub_view_size:int|list[int], 
                 dim:Literal["1d","2d","3d"], 
                 in_channels:int,
                 loss_weight:float=1., ):
        super().__init__()
        
        self.sub_view_size = np.array(
            [sub_view_size] * DIM_MAP[dim] 
            if isinstance(sub_view_size, int) 
            else sub_view_size)
        self.view_size = np.array(
            [view_size] * DIM_MAP[dim] 
            if isinstance(view_size, int) 
            else view_size)
        
        if len(self.sub_view_size) != DIM_MAP[dim]:
            raise ValueError(f"sub_volume_size length {len(self.sub_view_size)} does not match dim {dim}")
        if np.any(self.sub_view_size > self.view_size):
            raise ValueError("Sub-view size cannot be larger than view size")
        
        self.loss_weight = loss_weight
        self.dim = dim
        self.in_channels = in_channels
        self.encoder = nn.ModuleList([
            eval(f"nn.Conv{dim}")(
                in_channels=in_channels*(2**i), 
                out_channels=in_channels*(2**(i+1)), 
                kernel_size=4, 
                stride=2,
                padding=1,)
            for i in range(4)
        ])
        self.act = nn.GELU()
        self.avg_pool = GlobalAvgPool(dim)
        self.sim_cri = nn.CosineSimilarity()
        self.gt = torch.arange(4).float()[None, None]  # [1 (batch), 1 (num_pairs), 4]
        
        self._init_pair_mask()

    def _get_AdjDst_indices(self, abs_gap: Tensor, coords: Tensor) -> tuple:
        """
        Locate the adjacent pair and distant pair of two sub-view.
        The results contain view indexs and volume absolute coords.
        
        Args:
            abs_gap (Tensor): [N, num_views, num_views, coord-dims]
            coords (Tensor): [N, num_views, coord-dims]
            volume_shape (list[int]): [N, ...]
        Returns:
            tuple[list[list[tuple]], Tensor]: 
                - indices: [N, num_pairs, 4, index (tuple) (view_idx, *coord-dim-slices)]
                - centers: [N, num_pairs, 4, coord-dims]
        """
        # prepare for the calculation
        N, num_views, cdims = coords.shape
        pairs = [(i, j) for i in range(num_views) for j in range(i+1, num_views)]
        num_pairs = len(pairs)
        indices = [[None]*num_pairs for _ in range(N)]
        centers = torch.zeros(N, num_pairs, 4, cdims)
        
        def sub_area_center(direction:Tensor) -> Tensor:
            """
            Given the sub-view size and the direction vector.
            It is possible to determine the maximum center offset along the direction,
            before the sub-view touches the view's boundary.
            
            Args:
                direction (Tensor): [..., coord-dims]
            Return:
                offset (Tensor): [..., coord-dims]
            """
            # The absolute gap available for the center to move 
            # before touching the boundary on each dimension.
            avail_gap = (self.view_size - self.sub_view_size) / 2
            # Calculate the relative gap corresponding to direction vector
            rel_direction_ratio = avail_gap / direction
            # Determine the maximum relative ratio.
            # The minimum value of all dimensions are used,
            # because it will be the first dimension to hit the boundary.
            max_direction_ratio = rel_direction_ratio.abs().min(dim=0).values
            # Return the absolute maximum offset on each dimension.
            return max_direction_ratio * direction

        offsets = torch.round(sub_area_center(abs_gap.cpu())).int().numpy()

        for n in range(N):
            # Traverse all possible view pairs among all views.
            for idx_pair, (i, j) in enumerate(pairs):
                # Get the maximum offset of the pair.
                # offset = torch.round(sub_area_center(abs_gap[n, i, j])).int()
                offset = offsets[n, i, j]
                
                # Calculate the center of the adjacent pair and the distant pair.
                # Volume coordinate system here.
                centers[n, idx_pair, 0] = coords[n, i] + offset  # adj1
                centers[n, idx_pair, 1] = coords[n, j] - offset  # adj2
                centers[n, idx_pair, 2] = coords[n, i] - offset  # dst1
                centers[n, idx_pair, 3] = coords[n, j] + offset  # dst2
                
                # Calculate the sub-view index of the adjacent pair and the distant pair.
                # View coordinate system here. (namely patch)
                v_c = self.view_size//2  # view center
                sub_v_c = self.sub_view_size//2  # sub-view center
                # v_c +/- derterimines the sub-view center
                # and the center +/- sub_v_c determines the sub-view boundary,
                # then slice index is available to directly sample from view array.
                indices[n][idx_pair] = [
                    # adj1
                    (i, ) + tuple(slice(v_c[x] + offset[x] - sub_v_c[x], 
                                        v_c[x] + offset[x] + sub_v_c[x]
                            ) for x in range(cdims)),
                    # adj2
                    (j, ) + tuple(slice(v_c[x] - offset[x] - sub_v_c[x], 
                                        v_c[x] - offset[x] + sub_v_c[x]
                            ) for x in range(cdims)),
                    # dst1
                    (i, ) + tuple(slice(v_c[x] - offset[x] - sub_v_c[x], 
                                        v_c[x] - offset[x] + sub_v_c[x]
                            ) for x in range(cdims)),
                    # dst2
                    (j, ) + tuple(slice(v_c[x] + offset[x] - sub_v_c[x], 
                                        v_c[x] + offset[x] + sub_v_c[x]
                            ) for x in range(cdims)),
                ]

        return indices, centers

    def _sub_volume_selector(self, nir: Tensor, indices: list[list[tuple]]) -> Tensor:
        """根据indices从nir中选择子区域
        
        Args:
            nir (Tensor): [N, num_views, C, ...]
            indices (list[list[tuple]]): [N, num_pairs, 4, (view_idx, *slices)]
        Returns:
            Tensor: [N, num_pairs, 4, C, ...]
        """
        N = len(indices)
        num_pairs = len(indices[0])
        samples = []
        
        for n in range(N):
            pair_samples = []
            for idx_pair in range(num_pairs):
                # 每对视图有4个子区域
                sub_volumes = []
                for idx_area in range(4):
                    # 获取索引元组
                    index = indices[n][idx_pair][idx_area]
                    view_idx = index[0]  # 第一个元素是view索引
                    slices = index[1:]   # 剩余元素是空间维度的切片
                    
                    # 从nir中提取子区域
                    sub_volumes.append(nir[n, view_idx, :, *slices])
                
                # 堆叠当前pair的4个子区域
                pair_samples.append(torch.stack(sub_volumes, dim=0))  # [4, C, ...]
                
            # 堆叠当前样本的所有pairs
            samples.append(torch.stack(pair_samples, dim=0))  # [num_pairs, 4, C, ...]
        
        # 堆叠所有样本
        return torch.stack(samples, dim=0)  # [N, num_pairs, 4, C, ...]

    def forward(self, sub_vols:Tensor) -> Tensor:
        """
        Args: 
            sub_vols (Tensor): [N, num_pairs, 4, C, ...]
            
        Returns:
            encoded_vols (Tensor): Same with sub_vols.
        
        NOTE
            The third dimension 4 equals to [adj1, adj2, dist1, dist2]
        """
        # [N, num_pairs, 4 (adj1, adj2, dist1, dist2), C, ...]
        ori_shape = sub_vols.shape
        f = []
        
        for pair in range(sub_vols.size(1)):
            for chunk in range(sub_vols.size(2)):
                v = sub_vols[:, pair, chunk]
                for enc_layer in self.encoder:
                    v = enc_layer(v)
                    v = self.act(v)
                f.append(v)
        
        f = torch.stack(f, dim=1) # [N, num_pairs*4, C, ...]
        return f.reshape(*ori_shape[0:3], *f.shape[2:])  # [N, num_pairs, 4, C, ...]

    def loss(self, nir:Tensor, sub_area_indices:np.ndarray) -> dict[str, Tensor]:
        """
        Args:
            neural implicit representation (Tensor): [N, num_views, C, ...]
            abs_gap (Tensor): [N, num_views (start from), num_views (point to), coord-dim-length]
            coords (Tensor): [N, num_views, 3]
        Returns:
            dict[str, Tensor]: 包含邻近和远离子区域的相似度损失
        """
        # 获取子区域表示
        # [N, num_pairs, 4 (v_from_adj, v_to_adj, v_from_dist, v_to_dist), C, ...]
        sub_areas = self._sub_volume_selector(nir, sub_area_indices)
        
        # 编码子区域
        f = self.forward(sub_areas)  # [N, num_pairs, 4, C, ...]
        
        # 计算相似度损失
        adja_losses = []
        dist_losses = []
        for pair in range(f.size(1)):
            p = f[:, pair]  # [N, 4, C, ...]
            adja_sim_loss = self.sim_cri(p[:,0], p[:,1])  # 计算邻近对的相似度
            dist_sim_loss = self.sim_cri(p[:,2], p[:,3])  # 计算远离对的相似度
            adja_losses.append(adja_sim_loss)
            dist_losses.append(dist_sim_loss)
        
        # 计算平均损失
        adja_losses = torch.stack(adja_losses, dim=1).mean()  # [N, num_pairs] → scalar
        dist_losses = torch.stack(dist_losses, dim=1).mean()  # [N, num_pairs] → scalar
        
        return {
            "loss/sim_adja": -adja_losses * self.loss_weight,  # 邻近对应该更相似（高similarity）
            "loss/sim_dist": dist_losses * self.loss_weight,   # 远离对应该更不相似（低similarity）
        }
    
    def _init_pair_mask(self):
        """Initialize and cache the upper triangle mask for pair finding."""
        self.pair_mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
        self.pair_mask = self.pair_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        self.pair_rows, self.pair_cols = torch.where(self.pair_mask[0, 0])
    
    def _find_closest_farthest_pairs(self, f) -> tuple[Tensor, Tensor]:
        """
        Args:
            f (Tensor): [N, num_pairs, 4, C, ...]
            
        Returns:
            closest_pairs (Tensor): [N, num_pairs, 2]
            farthest_pairs (Tensor): [N, num_pairs, 2]
        """
        N, num_pairs, _, C, *rest = f.shape
        vectors = f.reshape(N, num_pairs, 4, -1)
        normalized = F.normalize(vectors, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(-2, -1))
        
        similarity_masked = similarity.cpu()
        valid_vals = similarity_masked[self.pair_mask.expand_as(similarity_masked)]
        valid_vals = valid_vals.view(N, num_pairs, -1)
        
        max_values, max_idx = valid_vals.max(dim=-1)
        min_values, min_idx = valid_vals.min(dim=-1)
        
        max_pairs = torch.stack([
            self.pair_rows[max_idx],
            self.pair_cols[max_idx]
        ], dim=-1)
        min_pairs = torch.stack([
            self.pair_rows[min_idx],
            self.pair_cols[min_idx]
        ], dim=-1)
        
        return max_pairs, min_pairs

    @torch.inference_mode()
    def predict(self, 
                nir:Tensor, 
                sub_view_indices:np.ndarray, 
                sub_view_coords:np.ndarray
    ) -> tuple:
        """
        Args:
            nir (Tensor): [N, num_views, C, ...]
            sub_view_indices (np.ndarray): [N, num_pairs, 4, index (tuple) (view_idx, *coord-dim-slices)]
            sub_view_coords (np.ndarray):  [N, num_pairs, 4, coord-dims]
            
        Returns:
            pred_pair_coords (Tensor): [N, num_pairs, 4, coord-dims]
            loss (Tensor): scalar
        """
        # [N, num_pairs, 4 (v_from_adj, v_to_adj, v_from_dist, v_to_dist), C, ...]
        sub_areas = self._sub_volume_selector(nir, sub_view_indices)
        
        f = self.forward(sub_areas)  # [N, num_pairs, 4, C, ...]
        
        # [N, num_pairs, 2], [N, num_pairs, 2]
        closest_pairs, farthest_pairs = self._find_closest_farthest_pairs(f)
        
        # [N, num_pairs, 4]
        pred_pair_idx = torch.cat([closest_pairs, farthest_pairs], dim=-1)
        
        loss = F.l1_loss(pred_pair_idx, self.gt.expand_as(pred_pair_idx))
        
        def gather_by_indices(source, indices):
            """
            resample on dim=2 from source according to indices
            
            source: [N, num_pairs, 4, coord_dims]
            indices: [N, num_pairs, 4]
            returns: [N, num_pairs, 4, coord_dims]
            """
            N, num_pairs, _, coord_dims = source.shape
            # [N, num_pairs, 4] -> [N, num_pairs, 4, 1]
            indices = indices[..., None]
            # [N, num_pairs, 4, 1] -> [N, num_pairs, 4, coord_dims]
            indices = indices.expand(-1, -1, -1, coord_dims)
            if isinstance(indices, Tensor):
                indices = indices.cpu().numpy()
            # execute gather
            return np.take_along_axis(source, indices, 2)
        
        # resample using sub_volume_indices
        pred_pair_coords = gather_by_indices(sub_view_coords, pred_pair_idx)
        assert pred_pair_coords.shape == sub_view_coords.shape

        return pred_pair_coords, loss


class VecAngConstraint(BaseVolumeWisePredictor):
    """
    Predict the absolute coordinations of views.
    The prediction is supervised by coordinations, and,
    the possible routes between the views.
    """
    
    def __init__(self, 
                 num_views:int, 
                 dim:Literal["1d","2d","3d"], 
                 loss_weight:float=1., 
                 *args, **kwargs):
        super().__init__(dim=dim, *args, **kwargs)
        self.loss_weight = loss_weight
        num_channel_from_super = self.channels[-1]
        self.proj_abs_loc = nn.Linear(num_channel_from_super, int(dim.replace('d','')))
        nn.init.ones_(self.proj_abs_loc.weight)
        self.cri = nn.SmoothL1Loss()
        self.cycle_route_index = torch.from_numpy(self.generate_all_cycles(num_views))  # [num_views, num_paths, path_steps, 2]

    def forward(self, nir:Tensor) -> Tensor:
        """
        Args:
            nir (Tensor): Size [N, num_views, C, Z, Y, X]
            rel_gap (Tensor): Size [N, num_views (start from), num_views (point to), coord-dim-length]
        
        Returns:
            vector gap sort loss (Tensor): [N, ]
        """
        nir = super().forward(nir)  # [N, num_views, C]
        dire_vect = self.proj_abs_loc(nir)  # [N, num_views, coord-dim-length]
        dire_vect_diff = dire_vect.unsqueeze(2) - dire_vect.unsqueeze(1)  # (N, num_views, num_views, C)
        return dire_vect_diff  # (N, num_views, num_views, C)

    def find_all_paths_batched(self, direction_tensor: Tensor) -> Tensor:
        B, L, _, D = direction_tensor.shape
        all_batch_paths = []
        
        for batch_idx in range(B):
            # 存储格式: {(start, end): [path_vectors]}
            paths_dict = {(i,j):[] for i in range(L) for j in range(L) if i != j}
            
            def backtrack(current: int, start: int, visited: set, path_vector: Tensor):
                if len(visited) == L:
                    paths_dict[(start, current)].append(path_vector)
                    return
                for next_point in range(L):
                    if next_point not in visited:
                        visited.add(next_point)
                        new_vector = path_vector + direction_tensor[batch_idx, current, next_point]
                        backtrack(next_point, start, visited, new_vector)
                        visited.remove(next_point)
            
            for start_point in range(L):
                backtrack(start_point, start_point, {start_point}, 
                        torch.zeros(D, device=direction_tensor.device))
            
            # 整理数据为张量形式 [L, L, max_paths, D]
            batch_paths = []
            for i in range(L):
                end_paths = []
                for j in range(L):
                    if i != j:
                        paths = paths_dict[(i,j)]
                        if paths:
                            end_paths.append(torch.stack(paths))
                        else:
                            end_paths.append(torch.zeros((1, D), device=direction_tensor.device))
                    else:
                        # 对角线位置(起点=终点)填充零向量
                        end_paths.append(torch.zeros((1, D), device=direction_tensor.device))
                
                # 确保每个终点的路径数量一致
                max_paths = max(p.shape[0] for p in end_paths)
                padded_end_paths = []
                for paths in end_paths:
                    if paths.shape[0] < max_paths:
                        padding = torch.zeros((max_paths - paths.shape[0], D), device=paths.device)
                        padded_end_paths.append(torch.cat([paths, padding], dim=0))
                    else:
                        padded_end_paths.append(paths)
                batch_paths.append(torch.stack(padded_end_paths))
            
            all_batch_paths.append(torch.stack(batch_paths))
        
        return torch.stack(all_batch_paths)  # [B, L, L, num_paths, D]

    @staticmethod
    def generate_all_cycles(n) -> np.ndarray:
        """
        Find all possible cycles from the given coords.
        
        Args:
            Number of Coordinates (int)
            
        Returns:
            found cycle paths (np.ndarray): [start_points, paths, path_steps, 2 (start_point_index, target_point_index)]
        """
        num_paths = math.factorial(n - 1)
        paths = np.zeros((n, num_paths, n, 2), dtype=int)
        
        for start in range(n):
            others = [i for i in range(n) if i != start]
            for path_idx, perm in enumerate(permutations(others)):
                # 第一个step
                paths[start, path_idx, 0] = [start, perm[0]]
                # 中间step
                for step in range(1, n-1):
                    paths[start, path_idx, step] = [perm[step-1], perm[step]]
                # for循环结束即完成了所有的step，再往下就会回到自身了
        
        return np.array(paths)

    @staticmethod
    def compute_cycle_gap_sum(gap:Tensor, cycle:Tensor):
        """
        Args:
            gap: [N, num_points, num_points, coord_dims]
            cycle: [num_points, num_paths, path_steps, 2]
                NOTE No batch dimension, 
                     because for the same number of points, 
                     the cycle is the same.
        Returns:
            cycle destination which should be zero, because all routes end at self.
                [N, num_points, num_paths, coord_dims]
        """
        s = cycle[..., 0]  # source indexs [num_points, num_paths, path_steps]
        t = cycle[..., 1]  # target indexs [num_points, num_paths, path_steps]
        route = gap[:, s, t, :]  # [N, num_points, num_paths, path_steps, coord_dims]
        # Offset to self. The optimal value should be zero.
        return route.sum(dim=-2)  # [N, num_points, num_paths, coord_dims]

    def loss(self, nir: Tensor, gap: Tensor) -> dict[str, Tensor]:
        """
        Args:
            nir (Tensor): [N, num_views, C, ...]
            gap (Tensor): [N, 3 (start from), 3 (point to), 3(coord-dim)]
        """
        dire_vect = self.forward(nir)  # [N, L, L, D]
        loss_vect = self.cri(dire_vect, gap)
        
        # [N, num_points, num_paths, coord_dims]
        route_pred_dest = self.compute_cycle_gap_sum(dire_vect, self.cycle_route_index)
        # [N, num_points, num_paths, coord_dims]
        route_gt_dest = self.compute_cycle_gap_sum(gap, self.cycle_route_index)
        loss_route = self.cri(route_pred_dest, route_gt_dest)
        
        return {"loss/vect": loss_vect * self.loss_weight, 
                "loss/route": loss_route * self.loss_weight}

    @torch.inference_mode()
    def predict(self, nir:Tensor, gap: Tensor|None=None) -> tuple[Tensor, Tensor|None]:
        """
        Args:
            nir (Tensor): [N, num_views, C, Z, Y, X]
            gap (Tensor): [N, 3 (start from), 3 (point to), 3(coord-dim)]
            
        Returns:
            vec_pred (Tensor): [N, num_views, num_views, 3] 
        """
        pred = self.forward(nir)
        if gap is not None:
            loss = self.cri(pred, gap)
        else:
            loss = None
        return pred, loss


class RelSim_Metric(BaseMetric):
    def __init__(self, 
                 collect_device: str = 'cpu', 
                 prefix: str = 'Perf'):
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples):
        """
        Args:
            data_batch: A batch of data from the dataloader.
            data_sample: datasample dict
                - volume: [C, ...]
                - sub_views: [num_views, C, ...]
                - nir (Neural Implicit Representation): [num_views, C, ...]
                - abs_gap (Absolute Gap): [num_views, num_views, 3 (coord-dims)]
                - rel_gap (Relative Gap): [num_views, num_views, 3 (coord-dims)]
                - gap_pred (Gap Prediction): [num_views, num_views]
                - sim_pred (Similarity): [num_pairs, 4 (i_adj1, i_adj2, i_dist1, i_dist2)]
                - vec_pred (Vector): [num_views, num_views, 3]
                - sim_chunk_size (Tensor): [num_pairs, 3]
        """
        for sample in data_samples:
            result = {
                "gap_loss": sample["gap_loss"],
                "sim_loss": sample["sim_loss"],
                "vec_loss": sample["vec_loss"],
            }
            self.results.append(result)

    def compute_metrics(self, results: list[dict]):
        """
        Args:
            results: A list of processed results.
        """
        c = lambda k, r: torch.stack([result[k] for result in r]).mean().cpu().numpy()
        context = {
            "gap_loss": c("gap_loss", results) if results else None,
            "sim_loss": c("sim_loss", results) if results else None,
            "vec_loss": c("vec_loss", results) if results else None,
        }
        context["all_loss"] = sum(context.values())
        return context


class RelSim_Viser(GeneralViser):
    def __init__(self, coord_norm:list[int], name:str|None=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.coord_norm = np.array(coord_norm)
    
    def _vis_gap(self, gap: Tensor, gt_gap: Tensor):
        """
        Args:
            gap (Tensor): [num_views, num_views]
            gt_gap (Tensor): [num_views, num_views, coord_dims]
        """
        gt_gap = gt_gap.norm(dim=-1)  # [num_views, num_views]
        
        # 获取上三角矩阵的掩码
        mask = torch.triu(torch.ones_like(gap), diagonal=1).bool()
        
        # 提取上三角矩阵的值
        gap_values = gap[mask].cpu().numpy()
        gt_gap_values = gt_gap[mask].cpu().numpy()
        
        # denorm
        gap_values *= np.linalg.norm(self.coord_norm)
        
        lower_bound = min(gap_values.min(), gt_gap_values.min())
        upper_bound = max(gap_values.max(), gt_gap_values.max())
        
        # 创建散点图
        plt.figure(figsize=(3, 3))
        plt.plot([lower_bound, upper_bound], 
                 [lower_bound, upper_bound], 
                 color=CMAP_COLOR[1], 
                 linestyle='--', 
                 zorder=0, 
                 alpha=0.7)
        plt.scatter(gap_values, gt_gap_values, color=CMAP_COLOR[0], alpha=0.6)
        
        # 设置图表属性
        plt.xlabel('Gap Prediction')
        plt.ylabel('Ground Truth')
        
        # to ndarray
        plt.tight_layout()
        img_arr = self._plt2array(plt.gcf())
        plt.close()
        return img_arr
    
    def _vis_sim(self, 
                entire_volume:np.ndarray, 
                views:Tensor, 
                view_coords:np.ndarray, 
                adja_coords:np.ndarray, 
                dist_coords:np.ndarray, 
                sim_chunk_size:np.ndarray):
        """
        A simple illustrative 3D example using matplotlib's 3D projection.
        It draws wireframes to approximate bounding boxes.
        Args:
            entire_volume (Tensor): [C, D, H, W]
            views (Tensor): [num_views, C, subD, subH, subW]
            view_coords (Tensor): [num_views, 3]
            adja_coords (Tensor): [num_pairs, 2, 3]
            dist_coords (Tensor): [num_pairs, 2, 3]
            sim_chunk_size (Tensor): [3 (coord_dims)]
        """
        views = views.cpu()
        
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        D, H, W = entire_volume.shape[1], entire_volume.shape[2], entire_volume.shape[3]
        # 禁用网格线
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"].update({
                "linestyle": ":", 
                "color": "gray", 
                "alpha": 0.1,
                "linewidth": 1})
        
        # 范围
        ax.set_xlim([0, W])
        ax.set_ylim([0, H])
        ax.set_zlim([0, D])
        # 刻度
        ax.set_xticks([0, W // 2, W])
        ax.set_yticks([0, H // 2, H])
        ax.set_zticks([0, D // 2, D])
        # 轴背景
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # 使用wireframe画一个bbox
        def draw_bbox(center, size, fill=False, fill_alpha=0.1, **kwargs):
            d, h, w = size
            z_c, y_c, x_c = center
            
            # bbox的八个顶点
            corners = []
            for dz in [ -d//2, d//2 ]:
                for dy in [ -h//2, h//2 ]:
                    for dx in [ -w//2, w//2 ]:
                        corners.append([
                            x_c + dx, 
                            y_c + dy, 
                            z_c + dz
                        ])
            corners = np.array(corners)
            
            # wireframe的12条边
            edges_idx = [
                (0,1), (0,2), (1,3), (2,3),
                (4,5), (4,6), (5,7), (6,7),
                (0,4), (1,5), (2,6), (3,7)
            ]
            for s,e in edges_idx:
                x_vals = [corners[s][0], corners[e][0]]
                y_vals = [corners[s][1], corners[e][1]]
                z_vals = [corners[s][2], corners[e][2]]
                ax.plot(x_vals, y_vals, z_vals, **kwargs)
            
            if fill:
                # 定义6个面的顶点索引
                faces_idx = [
                    [0,1,3,2],  # 前
                    [4,5,7,6],  # 后
                    [0,1,5,4],  # 下
                    [2,3,7,6],  # 上
                    [0,2,6,4],  # 左
                    [1,3,7,5]   # 右
                ]
                
                # 收集面的顶点
                faces = []
                for face in faces_idx:
                    faces.append([corners[idx] for idx in face])
                    
                # 创建面的集合并设置属性
                collection = art3d.Poly3DCollection(faces)
                collection.set_alpha(fill_alpha)
                collection.set_facecolor(kwargs.get('color', 'blue'))
                ax.add_collection3d(collection)

        # 绘制所有view
        _, _, vD, vH, vW = views.shape
        for coord in view_coords:
            draw_bbox(coord, 
                    (vD, vH, vW), 
                    fill=True, 
                    fill_alpha=0.1, 
                    color='black', 
                    linewidth=0.5, 
                    alpha=0.2, 
                    linestyle='--', 
                    zorder=3)
        
        # 绘制多对adja和dist chunk
        num_pairs = adja_coords.shape[0]
        for i in range(num_pairs):
            adja_centers = adja_coords[i]
            dist_centers = dist_coords[i]
            # 两个adja
            for ac in adja_centers:
                draw_bbox(
                    ac, 
                    sim_chunk_size, 
                    color=DEFAULT_CMAP(32), 
                    alpha=0.9, 
                    linewidth=1, 
                    zorder=2)
            # 两个dist
            for dc in dist_centers:
                draw_bbox(
                    dc, 
                    sim_chunk_size, 
                    color=DEFAULT_CMAP(224), 
                    alpha=0.9, 
                    linewidth=1.5, 
                    zorder=1)
        
        # 图例
        legend_elements = [
            Patch(facecolor=DEFAULT_CMAP(32), alpha=0.9, label='Adjacent Chunks'),
            Patch(facecolor=DEFAULT_CMAP(224), alpha=0.9, label='Distant Chunks'),
            Patch(facecolor='black', alpha=0.2, label='Sub Views')
        ]
        ax.legend(handles=legend_elements, 
                bbox_to_anchor=(0.25, 0.9),
                loc='center',
                frameon=False)
        
        def draw_surface(ax:Axes, entire_volume):
            C, Z, Y, X = entire_volume.shape
            front_slice = entire_volume[0, :, :, X//2]  # y-z平面
            top_slice   = entire_volume[0, Z//2, :, :]    # x-y平面
            right_slice = entire_volume[0, :, Y//2, :]  # x-z平面
            
            def normalize(x):
                return (x - x.min()) / (x.max() - x.min())
            
            front_slice = normalize(front_slice)
            top_slice = normalize(top_slice)
            right_slice = normalize(right_slice)
            
            y, z = np.meshgrid(np.linspace(0, H, H), np.linspace(0, D, D))
            x = np.full_like(y, W)
            ax.plot_surface(x, y, z, facecolors=plt.cm.gray(front_slice), alpha=0.3, edgecolor='none')
            
            x, y = np.meshgrid(np.linspace(0, W, W), np.linspace(0, H, H))
            z = np.full_like(x, D)
            ax.plot_surface(x, y, z, facecolors=plt.cm.gray(top_slice), alpha=0.3, edgecolor='none')
            
            x, z = np.meshgrid(np.linspace(0, W, W), np.linspace(0, D, D))
            y = np.full_like(x, H)
            ax.plot_surface(x, 0, z, facecolors=plt.cm.gray(right_slice), alpha=0.3, edgecolor='none')
        
        draw_surface(ax, entire_volume)
        
        plt.tight_layout()
        img_arr = self._plt2array(fig)
        plt.close(fig)
        return img_arr

    def _vis_vec(self, vec:Tensor, coords:np.ndarray, gt_vec:Tensor):
        """
        Args:
            coords (np.ndarray): [num_views, 3 (coord-dim)]
            vec (np.ndarray): [num_views, num_views, 3]
            gt_vec (np.ndarray): [num_views, num_views, 3]
        """
        vec = vec.cpu()
        gt_vec = gt_vec.cpu()
        
        # denorm
        vec *= torch.from_numpy(self.coord_norm)[None, None]
        
        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4,1])
        n = coords.shape[0]
        colors = [DEFAULT_CMAP(i/(n-1)) for i in range(n)]
        ax_main = fig.add_subplot(gs[0,0], projection='3d')

        # 主图坐标
        ax_main.scatter3D(
            coords[:,0], coords[:,1], coords[:,2], 
            s=70,
            marker='x', 
            c=colors,
            linewidths=2)  # 使用colormap
        ax_main.set_xlabel('X')
        ax_main.set_ylabel('Y')
        ax_main.set_zlabel('Z')

        # 主图向量
        for i in range(n-1):
            source = coords[i]
            pred = vec[i, i+1]
            ax_main.quiver(
                *source, *pred, 
                color=colors[i+1],  # 使用目标点的颜色
                alpha=0.8, 
                length=1, 
                arrow_length_ratio=0.1,
                linewidths=2)

        # 右侧三张子图
        def draw_subfig(gs: SubplotSpec, vec: Tensor, gt_vec: Tensor):
            """绘制三个维度上的预测vs实际差距散点图
            
            Args:
                gs (SubplotSpec): 子图网格
                vec (Tensor): 预测向量 [num_views, num_views, 3]
                abs_gap (Tensor): 实际差距 [num_views, num_views, 3]
            """
            sub_gs = gs.subgridspec(nrows=3, ncols=1)
            axes = [plt.subplot(sub_gs[i]) for i in range(3)]
            dim_names = ['X', 'Y', 'Z']
            
            n = vec.shape[0]
            triu_indices = torch.triu_indices(n, n, offset=1)
            for dim in range(3):
                ax = axes[dim]
                pred = vec[triu_indices[0], triu_indices[1], dim].cpu().numpy()
                gt = gt_vec[triu_indices[0], triu_indices[1], dim].cpu().numpy()
                
                # 散点图
                ax.scatter(pred, gt, alpha=0.5, s=20, color=CMAP_COLOR[1])
                # 对角线
                lims = [
                    min(pred.min(), gt.min()),
                    max(pred.max(), gt.max()),
                ]
                ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0)

                ax.set_title(f'{dim_names[dim]}')
                ax.set_aspect('equal')
        
        # 绘制子图
        draw_subfig(gs[0,1], vec, gt_vec)

        fig.tight_layout()
        img_arr = self._plt2array(fig)
        plt.close(fig)
        return img_arr

    @master_only
    def add_datasample(self, data_sample:DataSample, step:int|None=None):
        """
        Args:
            data_sample: datasample dict
                - volume: [C, ...]
                - sub_views: [num_views, C, ...]
                - coords: [num_views, 3]
                - abs_gap (Absolute Gap): [num_views, num_views, 3 (coord-dims)]
                - rel_gap (Relative Gap): [num_views, num_views, 3 (coord-dims)]
                - nir (Neural Implicit Representation): [num_views, C, ...]

                - gap_pred (Gap Prediction): [num_views, num_views]
                - sim_pred_coords (Similarity): [num_pairs, 4 (i_adj1, i_adj2, i_dist1, i_dist2), 3]
                - vec_pred (Vector): [num_views, num_views, 3]
                - sim_chunk_size (Tensor): [num_pairs, 3]
        """
        gap_vis_img = self._vis_gap(data_sample.gap_pred, 
                                    data_sample.abs_gap)
        sim_vis_img = self._vis_sim(data_sample.volume, 
                                    data_sample.sub_views, 
                                    data_sample.view_coords, 
                                    data_sample.sim_pred[:, :2], 
                                    data_sample.sim_pred[:, 2:], 
                                    data_sample.sim_chunk_size)
        vec_vis_img = self._vis_vec(data_sample.vec_pred,
                                    data_sample.view_coords,
                                    data_sample.abs_gap)
        
        img_name = os.path.basename(data_sample.img_path)
        self.add_image(f'PredImg_{img_name}/Gap', gap_vis_img, step)
        self.add_image(f'PredImg_{img_name}/Sim', sim_vis_img, step)
        self.add_image(f'PredImg_{img_name}/Vec', vec_vis_img, step)
