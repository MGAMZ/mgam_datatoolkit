import os
import pdb
import pytest
from abc import abstractmethod
from functools import partial
from typing_extensions import Literal, OrderedDict, Sequence
from itertools import product

import numpy as np
import torch
import mpl_toolkits.mplot3d.art3d as art3d
from torch import Tensor
from torch.nn import PixelUnshuffle as PixelUnshuffle2D
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.axes import Axes
from matplotlib.gridspec import SubplotSpec

from mmcv.transforms import BaseTransform
from mmengine.config import ConfigDict
from mmengine.runner import Runner
from mmengine.model import MomentumAnnealingEMA, BaseModule
from mmengine.utils.misc import is_list_of
from mmengine.dist import all_gather, get_rank, is_main_process, master_only
from mmengine.visualization import Visualizer
from mmengine.hooks import Hook
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup.base import BaseSelfSupervisor
from mmpretrain.models.selfsup.mocov3 import CosineEMA

from mgamdata.mm.mmseg_Dev3D import PixelUnshuffle1D, PixelUnshuffle3D


DIM_MAP = {"1d": 1, "2d": 2, "3d": 3}
CMAP_SEQ_COLOR = ["winter", "RdPu"]
DEFAULT_CMAP = plt.get_cmap(CMAP_SEQ_COLOR[0])
CMAP_COLOR = [DEFAULT_CMAP(32), DEFAULT_CMAP(224)]


class MoCoV3Head_WithAcc(BaseModule):
    def __init__(
        self,
        embed_dim: int,
        proj_channel: int,
        dim: Literal["1d", "2d", "3d"],
        loss: dict,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_channel = proj_channel
        self.dim = dim
        self.loss_module = MODELS.build(loss)
        self.temperature = temperature
        self.down_r = 4
        self.predictor = self._init_proj()
        self.target_proj = self._init_proj()

    def _init_proj(self):
        if self.dim == "1d":
            proj_conv = torch.nn.Conv1d
            avgpool = partial(torch.nn.AdaptiveAvgPool1d, output_size=(1))
            pus = PixelUnshuffle1D
        elif self.dim == "2d":
            proj_conv = torch.nn.Conv2d
            avgpool = partial(torch.nn.AdaptiveAvgPool2d, output_size=(1, 1))
            pus = PixelUnshuffle2D
        elif self.dim == "3d":
            proj_conv = torch.nn.Conv3d
            avgpool = partial(torch.nn.AdaptiveAvgPool3d, output_size=(1, 1, 1))
            pus = PixelUnshuffle3D
        else:
            raise NotImplementedError(f"Invalid Dim Setting: {self.dim}")

        return torch.nn.Sequential(
            pus(downscale_factor=self.down_r),  # C_out = factor**dim * C_in
            proj_conv(
                self.down_r ** int(self.dim[0]) * self.embed_dim, self.proj_channel, 1
            ),
            torch.nn.GELU(),
            avgpool(),
            torch.nn.Flatten(start_dim=1),
        )

    def loss(
        self, base_out: Tensor, momentum_out: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate loss.

        Args:
            base_out (Tensor): [N, C, ...] features from base_encoder.
            momentum_out (Tensor): [N, C, ...] features from momentum_encoder.

        Returns:
            Tensor: The loss tensor.
        """
        # predictor computation
        pred = self.predictor(base_out)  # NxC
        target = self.target_proj(base_out)  # NxC

        # normalize
        pred = torch.nn.functional.normalize(pred, dim=1)
        target = torch.nn.functional.normalize(target, dim=1)

        # get negative samples
        target = torch.cat(all_gather(target), dim=0)

        # Einstein sum is more intuitive
        logits = torch.einsum("nc,mc->nm", [pred, target]) / self.temperature

        """
        使用一个混淆矩阵来表达经过两组不同的变换之后的同batch样本之间的相似度
        理想情况下，模型应当能识别出同样的样本，因此这个矩阵应当是对角线上有较大值，其他地方为较小值
        从分类任务混淆矩阵的角度出发，这代表着样本的gt标签就是它们自身的index
        """

        # generate labels
        batch_size = logits.shape[0]
        labels = (
            torch.arange(batch_size, dtype=torch.long) + batch_size * get_rank()
        ).to(logits.device)

        loss = self.loss_module(logits, labels)
        return loss, logits, labels


class ReconstructionHead(BaseModule):
    def __init__(
        self,
        model_out_channels: int,
        recon_channels: int,
        dim: Literal["1d", "2d", "3d"],
        reduction: str = "mean",
        loss_type: Literal["L1", "L2"] = "L1",
    ):
        super().__init__()
        self.model_out_channels = model_out_channels
        self.recon_channels = recon_channels
        self.loss_type = loss_type
        self.dim = dim
        self.criterion = (
            torch.nn.L1Loss(reduction=reduction)
            if loss_type == "L1"
            else torch.nn.MSELoss(reduction=reduction)
        )
        self.conv_proj = eval(f"torch.nn.Conv{dim}")(
            model_out_channels, recon_channels, 1
        )

    def loss(self, recon: Tensor, ori: Tensor):
        proj = self.conv_proj(recon)
        loss = self.criterion(proj.squeeze(), ori.squeeze())
        return {f"loss_recon_{self.loss_type}": loss, "reconed": proj}


class AutoEncoderSelfSup(BaseSelfSupervisor):
    def __init__(
        self,
        encoder: dict,
        neck: dict | None = None,
        decoder: dict | None = None,
        head: dict | None = None,
        pretrained: str | None = None,
        data_preprocessor: dict | None = None,
        init_cfg: list[dict] | dict | None = None,
        *args,
        **kwargs,
    ) -> None:
        encoder_decoder = torch.nn.Sequential(
            MODELS.build(encoder),
            MODELS.build(neck) if neck is not None else torch.nn.Identity(),
            MODELS.build(decoder) if decoder is not None else torch.nn.Identity(),
        )
        super().__init__(
            backbone=encoder_decoder,
            neck=None,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            *args,
            **kwargs,
        )

    @property
    def whole_model_(self) -> torch.nn.Module:
        if self.with_neck:
            return torch.nn.Sequential(self.backbone, self.neck)
        else:
            return self.backbone

    def parse_losses(
        self,
        losses: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        log_vars = []
        for loss_name, loss_value in losses.items():
            if "loss" in loss_name:
                if isinstance(loss_value, torch.Tensor):
                    log_vars.append([loss_name, loss_value.mean()])
                elif is_list_of(loss_value, torch.Tensor):
                    log_vars.append(
                        [loss_name, sum(_loss.mean() for _loss in loss_value)]
                    )
                else:
                    raise TypeError(f"{loss_name} is not a tensor or list of tensors")
            else:
                log_vars.append([loss_name, loss_value])

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)  # type: ignore
        return loss, log_vars  # type: ignore

    @abstractmethod
    def loss(
        self, inputs: list[Tensor], data_samples: list[DataSample], **kwargs
    ) -> dict[str, Tensor]: ...


class AutoEncoder_MoCoV3(AutoEncoderSelfSup):
    def __init__(self, base_momentum: float = 0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_momentum = base_momentum
        self.momentum_encoder = CosineEMA(
            self.whole_model_, momentum=base_momentum
        )

    @staticmethod
    def calc_acc(logits: Tensor, labels: Tensor) -> Tensor:
        """Calculate the accuracy of the model.

        Args:
            logits (Tensor): The output logits, shape (N, C).
            labels (Tensor): The target labels, shape (N).

        Returns
            Tensor: The accuracy of the model.
        """
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / labels.shape[0]
        return acc.unsqueeze(0)

    def loss(
        self, inputs: list[Tensor], data_samples: list[DataSample], **kwargs
    ) -> dict[str, Tensor]:
        """The forward function in training.

        Args:
            inputs (List[Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(inputs, list)
        self.backbone: BaseModule
        self.neck: BaseModule
        self.head: BaseModule

        q1 = self.backbone(inputs[0])[0]
        q2 = self.backbone(inputs[1])[0]

        # compute key features, [N, C] each, no gradient
        with torch.no_grad():
            # update momentum encoder
            self.momentum_encoder.update_parameters(self.whole_model_)

            k1 = self.momentum_encoder(inputs[0])[0]
            k2 = self.momentum_encoder(inputs[1])[0]

        selfsup1 = self.head.loss(q1, k2)
        selfsup2 = self.head.loss(q2, k1)

        loss = selfsup1[0] + selfsup2[0]
        acc1 = self.calc_acc(logits=selfsup1[1], labels=selfsup1[2])
        acc2 = self.calc_acc(logits=selfsup2[1], labels=selfsup2[2])
        acc = (acc1 + acc2) / 2
        acc = torch.cat(all_gather(acc)).mean()
        losses = dict(loss_MoCoV3=loss, acc_MoCoV3=acc)
        return losses


class AutoEncoder_Recon(AutoEncoderSelfSup):
    def loss(
        self, inputs: list[Tensor], data_samples: list[DataSample], **kwargs
    ) -> dict[str, Tensor]:
        """The forward function in training.

        Args:
            inputs (List[Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(inputs, list)
        self.backbone: BaseModule
        self.head: BaseModule
        losses = {}
        recon = self.backbone(inputs[0])[0]
        ori = inputs[1]
        selfsup_loss = self.head.loss(recon, ori)

        losses.update(selfsup_loss)
        return losses


class RandomSubView(BaseTransform):
    """
        Get random sub-view from the original image.
        
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
    def __init__(self, num_views: int, dim: Literal["1d", "2d", "3d"], size: tuple[int]):
        self.num_views = num_views
        self.dim = dim
        self.size = size
    
    def _determine_slices(self, shape: tuple[int]) -> tuple[list[slice], list[int]]:
        full_slices = [slice(None)] * len(shape)
        center_coords = []
        
        for i, s in enumerate(self.size):
            dim_idx = -(i + 1)
            start = np.random.randint(0, shape[dim_idx] - s)
            full_slices[dim_idx] = slice(start, start + s)
            center_coords.insert(0, start + s // 2)
            
        return full_slices, center_coords
    
    def _get_sub_view(self, array: np.ndarray, slices: list[slice]) -> np.ndarray:
        return array[tuple(slices)]
    
    def transform(self, results):
        img = results["volume"] = results["img"]
        segs = {seg_field: [] for seg_field in results.get("seg_fields", [])}
        coords = []
        img_views = []
        
        for _ in range(self.num_views):
            slices, center_coord = self._determine_slices(img.shape)
            
            sub_img_view = self._get_sub_view(img, slices)
            img_views.append(sub_img_view)
            coords.append(center_coord)
            
            for seg_field in results.get("seg_fields", []):
                seg_slices = slices[1:] if len(slices) > 1 else slices
                sub_seg = self._get_sub_view(results[seg_field], seg_slices)
                segs[seg_field].append(sub_seg)
        
        results["img"] = np.stack(img_views)  # [num_views, *]
        # [num_views, num_spatial_dims]
        results["view_coords"] = torch.from_numpy(np.array(coords)).float()
        for seg_field in results.get("seg_fields", []):
            results[seg_field] = np.stack(segs[seg_field])
        
        return results


class NormalizeCoord(BaseTransform):
    """
        Required fields:
            - view_coords: [num_views, num_spatial_dims]
        
        Modified fields:
            - view_coords: [num_views, num_spatial_dims]
    """
    def __init__(self, div: list[int]):
        self.div = div
    
    def transform(self, results):
        coords = results["view_coords"]
        for i, s in enumerate(self.div):
            coords[:, i] = coords[:, i] / self.div[i]
        results["view_coords"] = coords
        return results


class RelativeSimilaritySelfSup(AutoEncoderSelfSup):
    def __init__(
        self, 
        gap_head:ConfigDict, 
        sim_head:ConfigDict, 
        vec_head:ConfigDict, 
        momentum=1e-4, 
        gamma=100, 
        update_interval=1, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gap_head:GapPredictor          = MODELS.build(gap_head)
        self.sim_head:SimPairDiscriminator  = MODELS.build(sim_head)
        self.vec_head:VecAngConstraint      = MODELS.build(vec_head)
        self.momentum_encoder = MomentumAnnealingEMA(
            self.whole_model_,
            momentum=momentum,
            gamma=gamma,
            interval=update_interval,
        )

    def parse_target(self, data_samples: list[DataSample]):
        # coords (coordinates):        [N, 3 (sub-volume), 3 (coord-dim)]
        coords = torch.stack([sample.view_coords for sample in data_samples])
        # abs_gap (absolute distance): [N, 3 (start from), 3 (point to), 3(coord-dim)]
        abs_gap = coords.unsqueeze(1) - coords.unsqueeze(2)
        # rel_gap (relative distance): [N, 3 (start from), 3 (point to), 3(coord-dim)]
        # The gap matrix is symmetric, so we can use the upper triangle part.
        # The following implementation is a trick, which will get relative value 
        # when comparing with the max gap value on each dimension.
        rel_base = abs_gap.max(dim=1).values.max(dim=1).values  # determine the max gap for each dimension
        rel_gap = abs_gap / rel_base[:, None, None, ...]  # normalize the gap matrix
        return coords, abs_gap, rel_gap

    def extract_nir(self, sv_main: Tensor, sv_aux:Tensor) -> Tensor:
        """
        Args:
            nir_main (Tensor): [N, C, Z, Y, X]
            nir_aux (Tensor): [N, sub-view, C, Z, Y, X]
        
        Returns:
            nir (Tensor): [N, sub-view, C, Z, Y, X]
        """
        # sv: sub view
        nir_main = self.whole_model_(sv_main)[0]
        # nir_sv: neural implicit representation of sub-sub_view
        nir_aux = [self.momentum_encoder(sub_view)[0] for sub_view in sv_aux.transpose(0, 1)]
        nir = torch.stack([nir_main, *nir_aux], dim=1)  # [N, sub_view, C, ...]
        return nir  # [N, sub-view, C, ...]

    def loss(
        self, inputs: Tensor, data_samples: list[DataSample], **kwargs
    ) -> dict[str, Tensor]:
        """
        Args:
            inputs (Tensor): [N, sub-view, C, *]
            data_samples (list[DataSample]): 
                DataSample:
                    - view_coords (Tensor): [sub-view, 3]
        """
        self.backbone: BaseModule
        self.head: BaseModule
        
        # sv: sub view
        sv_main = inputs[:, 0]
        sv_aux = inputs[:, 1:]
        coords, abs_gap, rel_gap = self.parse_target(data_samples)
        
        # neural implicit representation forward
        nir = self.extract_nir(sv_main, sv_aux)  # [N, sub-view, C, ...]
        
        losses = {}
        # relative gap self-supervision
        gap_losses = self.gap_head.loss(nir, abs_gap)
        for k, v in gap_losses.items():
            losses[k] = v
        # similarity self-supervision (Opposite nir has larger difference, namely negative label)
        sim_losses = self.sim_head.loss(nir, abs_gap, coords)
        for k, v in sim_losses.items():
            losses[k] = v
        # vector self-supervision
        vec_losses = self.vec_head.loss(nir, abs_gap)
        for k, v in vec_losses.items():
            losses[k] = v
        
        # update momentum model
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
                    - view_coords (Tensor): [sub-view, 3]
                    - volume (np.ndarray): [C, ...]
        
        Returns:
            list[DataSample]:
                DataSample:
                    - volume (np.ndarray): [sub-view, C, ...]
                    - rel_gap_gt (Tensor): [sub-view (start from), sub-view (point to), 3]
                    - abs_gap_gt (Tensor): [sub-view (start from), sub-view (point to), 3]
                    - coords_gt (Tensor): [sub-view, 3]
                    - view_coords (Tensor): [sub-view, 3]
                    - gap_pred (Tensor): [sub-view, sub-view]
                    - sim_pred (Tensor): [sub-view, sub-view]
                    - vec_pred (Tensor): [sub-view, sub-view, 3]
        """
        coords, abs_gap, rel_gap = self.parse_target(data_samples)

        sv_main = inputs[:, 0]
        sv_aux = inputs[:, 1:]
        # neural implicit representation forward
        nir = self.extract_nir(sv_main, sv_aux)  # [N, sub-view, C, ...]

        # [N, sub-view, sub-view]
        gap_pred, gap_loss = self.gap_head.predict(nir)
        # [N, num_pairs, 4 (i_adj1, i_adj2, i_dist1, i_dist2)]
        sim_pred, sim_loss = self.sim_head.predict(nir, abs_gap)
        # [N, sub-view, sub-view, 3]
        vec_pred, vec_loss = self.vec_head.predict(nir)

        for i in range(len(data_samples)):
            data_samples[i].sub_views = inputs[i]
            data_samples[i].nir = nir[i]
            data_samples[i].rel_gap_gt = rel_gap[i]
            data_samples[i].abs_gap_gt = abs_gap[i]
            data_samples[i].coords = coords[i]
            data_samples[i].gap_pred = gap_pred[i]
            data_samples[i].sim_pred = sim_pred[i]
            data_samples[i].vec_pred = vec_pred[i]
            data_samples[i].sim_chunk_size = self.sim_head.s
            data_samples[i].gap_loss = gap_loss
            data_samples[i].sim_loss = sim_loss
            data_samples[i].vec_loss = vec_loss

        return data_samples
    
    def forward(self,
                inputs: torch.Tensor|list[torch.Tensor],
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


class GlobalAvgPool(torch.nn.Module):
    
    def __init__(self, dim: Literal["1d", "2d", "3d"]):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == "1d":
            return torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        elif self.dim == "2d":
            return torch.nn.functional.adaptive_avg_pool2d(x, (1,1)).squeeze(-1).squeeze(-1)
        elif self.dim == "3d":
            return torch.nn.functional.adaptive_avg_pool3d(x, (1,1,1)).squeeze(-1).squeeze(-1).squeeze(-1)
        else:
            raise NotImplementedError(f"Invalid Dim Setting: {self.dim}")


class BaseVolumeWisePredictor(BaseModule):
    def __init__(self, dim:Literal["1d","2d","3d"], in_channels:int, num_views:int=3):
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.act = torch.nn.GELU()
        self.proj = torch.nn.ModuleList([
            eval(f"torch.nn.Conv{dim}")(in_channels//(2**i), in_channels//(2**(i+1)), 3, 2)
            for i in range(3)
        ])
        self.avg_pool = GlobalAvgPool(dim)
    
    def forward(self, x:Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, num_views, C, Z, Y, X]
        
        Returns:
            x (Tensor): [N, num_views, C]
        """
        x = x.transpose(1,0)
        extreacted = []
        for batch in x:
            for proj in self.proj:
                batch = proj(batch)
                batch = self.act(batch)
            batch = self.avg_pool(batch)
            extreacted.append(batch)
        
        return torch.stack(extreacted, dim=1)  # [N, num_views, C]


class GapPredictor(BaseVolumeWisePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cri = torch.nn.SmoothL1Loss()
    
    def forward(self, nir:Tensor) -> Tensor:
        """
        Args:
            nir (Tensor): Size [N, num_views, C, Z, Y, X]
            rel_gap (Tensor): Size [N, num_views (start from), num_views (point to), coord-dim-length]
        
        Returns:
            vector gap sort loss (Tensor): [N, ]
        """
        nir = super().forward(nir)  # [N, num_views, C]
        
        # Relative Positional Representation of Each Sub-Volume
        # The origin may align with the mean value of all samples' world coordinate systems'origin.
        # diff equals to the relative distance between each `nir`.
        rel_pos_rep_diff = nir.unsqueeze(2) - nir.unsqueeze(1)  # (N, num_views, num_views, C)
        # calculate the distance of `rel_pos_rep_diff`
        similarity = rel_pos_rep_diff.norm(dim=-1)  # (N, num_views, num_views)
        
        return similarity  # (N, num_views, num_views)
    
    def loss(self, nir:Tensor, rel_gap:Tensor) -> dict[str, Tensor]:
        """
        Args:
            nir (Tensor): Size [N, num_views, C, ...]
            rel_gap (Tensor): Size [N, num_views (start from), num_views (point to), coord-dim-length]
        """
        # (N, num_views, num_views)
        similarity = self.forward(nir)
        # (N, num_views, num_views)
        loss = self.cri(similarity, rel_gap.norm(dim=-1))
        return {"loss_gap": loss}

    @torch.inference_mode()
    def predict(self, nir:Tensor, rel_gap:Tensor|None=None) -> tuple[Tensor, Tensor|None]:
        """
        Args:
            nir (Tensor): [N, num_views, C, ...]
            rel_gap (Tensor): Size [N, num_views (start from), num_views (point to), coord-dim-length]
            
        Returns:
            gap_pred (Tensor): [N, num_views, num_views]
        """
        pred = self.forward(nir)
        if rel_gap is not None:
            loss = self.cri(pred, rel_gap.norm(dim=-1))
        else:
            loss = None
        return pred, loss


class SimPairDiscriminator(BaseModule):
    LABEL_ADJA_PAIR = 0
    LABEL_DIST_PAIR = 1
    
    def __init__(self, sub_view_size:int|list[int], dim:Literal["1d","2d","3d"], in_channels:int):
        super().__init__()
        # 统一sub_volume_size格式并根据dim检查维度
        
        self.s = ([sub_view_size] * DIM_MAP[dim] if isinstance(sub_view_size, int) 
                 else sub_view_size)
        
        if len(self.s) != DIM_MAP[dim]:
            raise ValueError(f"sub_volume_size length {len(self.s)} does not match dim {dim}")
            
        self.dim = dim
        self.in_channels = in_channels
        torch.nn.Conv1d
        self.encoder = torch.nn.ModuleList([
            eval(f"torch.nn.Conv{dim}")(
                in_channels=in_channels*(2**i), 
                out_channels=in_channels*(2**(i+1)), 
                kernel_size=4, 
                stride=2,
                padding=1,)
            for i in range(4)
        ])
        self.act = torch.nn.GELU()
        self.avg_pool = GlobalAvgPool(dim)
        self.sim_cri = torch.nn.CosineSimilarity()
        self.gt = torch.arange(4).float()[None, None]  # [1 (batch), 1 (num_pairs), 4]
    
    def _get_subvolume_indices(self, abs_gap:Tensor, coords:Tensor, volume_shape:list[int]):
        """
        Args:
            abs_gap (Tensor): [N, num_views (start from), num_views (point to), coord-dims]
            coords (Tensor): [N, num_views, 3]
            volume_shape (list[int]): [N, ...]
        Returns:
            indices (list[dict]): [N, num_pairs, 4, *slice]
        """
        

    def _sub_volume_selector(self, nir:Tensor, sub_volume_indices):
        """
        Args:
            nir (Tensor): [N, num_views, C, ...]
            sub_volume_indices (list[dict]): [N, num_pairs, 3]
        Returns:
            Samples (Tensor): [N, num_pairs, 4, C, ...]
        """
        batched_samples = []
        for n in range(len(sub_volume_indices)):
            processed_pairs = set()
            samples = []
            coords = []
            
            for (vol_from, vol_to) in sub_volume_indices[n].keys():
                if (vol_from, vol_to) in processed_pairs or (vol_to, vol_from) in processed_pairs:
                    continue
                    
                # fetch index
                _, adj_from_to, dist_from_to, coord_adj_from_to, coord_dst_from_to = sub_volume_indices[n][(vol_from, vol_to)]
                _, adj_to_from, dist_to_from, coord_adj_to_from, coord_dst_to_from = sub_volume_indices[n][(vol_to, vol_from)]
                
                # select adjacent sub-nir
                v_from_adj = nir[n, vol_from, :, *adj_from_to]
                v_to_adj = nir[n, vol_to, :, *adj_to_from]
                
                # select distant sub-nir
                v_from_dist = nir[n, vol_from, :, *dist_from_to]
                v_to_dist = nir[n, vol_to, :, *dist_to_from]
                
                # sample: [4, C, ...]
                sample = torch.stack([v_from_adj, v_to_adj, v_from_dist, v_to_dist], dim=0)
                samples.append(sample)
                # coord: [4, 3]
                coord = torch.stack([coord_adj_from_to, coord_dst_from_to, coord_adj_to_from, coord_dst_to_from], dim=0)
                
                processed_pairs.add((vol_from, vol_to))  # mark as processed
            
            batched_samples.append(torch.stack(samples, dim=0)) # [num_pairs, 4, C, ...]
            
        return torch.stack(batched_samples, dim=0)  # [N, num_pairs, 4, C, ...]

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

    def loss(self, nir:Tensor, abs_gap:Tensor, coords:Tensor,) -> dict[str, Tensor]:
        """
        Args:
            neural implicit representation (Tensor): [N, num_views, C, ...]
            abs_gap (Tensor): [N, num_views (start from), num_views (point to), coord-dim-length]
            coords (Tensor): [N, num_views, 3]
        """
        # determine the adjacent and distant chunks' position
        sub_volume_indices, _ = self._get_subvolume_indices(abs_gap, [nir.shape[0], *nir.shape[3:]])
        # get view of these positions 
        # [N, num_pairs, 4 (v_from_adj, v_to_adj, v_from_dist, v_to_dist), C, ...]
        sub_volumes = self._sub_volume_selector(nir, sub_volume_indices)
        
        f = self.forward(sub_volumes)  # [N, num_pairs, 4, C, ...]
        
        # calculate similarity loss
        adja_losses = []
        dist_losses = []
        for pair in range(f.size(1)):
            p = f[:, pair]  # [N, 4, C, ...]
            adja_sim_loss = self.sim_cri(p[:,0], p[:,1])
            dist_sim_loss = self.sim_cri(p[:,2], p[:,3])
            adja_losses.append(adja_sim_loss)
            dist_losses.append(dist_sim_loss)
        adja_losses = torch.stack(adja_losses, dim=1).mean()  # mean of [N, num_pairs]
        dist_losses = torch.stack(dist_losses, dim=1).mean()  # mean of [N, num_pairs]
        
        return {"loss_sim_adja": -adja_losses,
                "loss_sim_dist": dist_losses,}

    def _find_closest_farthest_pairs(self, f) -> tuple[Tensor, Tensor]:
        """
        Args:
            f (Tensor): [N, num_pairs, 4, C, ...]
            
        Returns:
            closest_pairs (Tensor): [N, num_pairs, 2]
            farthest_pairs (Tensor): [N, num_pairs, 2]
        """
        
        N, num_pairs, _, C, *rest = f.shape
        vectors = f.reshape(N, num_pairs, 4, -1)  # [N, num_pairs, 4, C*...]
        
        # cosine similarity matrix
        normalized = F.normalize(vectors, dim=-1)  # [N, num_pairs, 4, C*...]
        similarity = torch.matmul(normalized, normalized.transpose(-2, -1))  # [N, num_pairs, 4, 4]
        
        # avoid self-similarity
        mask = torch.eye(4, device=f.device)[None, None, :, :]
        similarity = similarity.masked_fill(mask.bool(), float('-inf'))
        
        # adjacent index
        values, indices = similarity.reshape(N, num_pairs, -1).sort(dim=-1, descending=True)
        closest_pairs = torch.div(indices[:, :, :2], 4, rounding_mode='floor')  # 取最高的两个相似度对应的行索引
        
        # distant index
        farthest_pairs = torch.div(indices[:, :, -2:], 4, rounding_mode='floor')  # 取最低的两个相似度对应的行索引

        # [N, num_pairs, 2], [N, num_pairs, 2]
        return closest_pairs, farthest_pairs

    @torch.inference_mode()
    def predict(self, nir:Tensor, abs_gap:Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            nir (Tensor): [N, num_views, C, ...]
            abs_gap (Tensor): [N, num_views, num_views, 3]
            
        Returns:
            sim_pred (Tensor): [N, num_pairs, 4 (adj1, adj2, dist1, dist2)]
        """
        sub_volume_indices, sub_volume_centers = self._get_subvolume_indices(
            abs_gap, [nir.shape[0], *nir.shape[3:]])
        sub_volumes = self._sub_volume_selector(nir, sub_volume_indices)
        
        f = self.forward(sub_volumes)  # [N, num_pairs, 4, C, ...]
        closest_pairs, farthest_pairs = self._find_closest_farthest_pairs(f)
        
        # [N, num_pairs, 4]
        pred_index = torch.cat([closest_pairs, farthest_pairs], dim=-1)
        # index from sub_volume_indices
        
        
        loss = F.l1_loss(pred_index, self.gt.cuda())
        return pred_index, loss


class VecAngConstraint(BaseVolumeWisePredictor):
    def __init__(self, dim:Literal["1d","2d","3d"], *args, **kwargs):
        super().__init__(dim=dim, *args, **kwargs)
        num_channel_from_super = self.proj[-1].out_channels
        self.proj_direction_vector = torch.nn.Linear(num_channel_from_super, int(dim.replace('d','')))
        torch.nn.init.ones_(self.proj_direction_vector.weight)
        self.cri = torch.nn.SmoothL1Loss()

    def forward(self, nir:Tensor) -> Tensor:
        """
        Args:
            nir (Tensor): Size [N, num_views, C, Z, Y, X]
            rel_gap (Tensor): Size [N, num_views (start from), num_views (point to), coord-dim-length]
        
        Returns:
            vector gap sort loss (Tensor): [N, ]
        """
        nir = super().forward(nir)  # [N, num_views, C]
        dire_vect = self.proj_direction_vector(nir)  # [N, num_views, coord-dim-length]
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

    def loss(self, nir: Tensor, abs_gap: Tensor) -> dict[str, Tensor]:
        """
        Args:
            nir (Tensor): [N, num_views, C, ...]
            abs_gap (absolute distance): [N, 3 (start from), 3 (point to), 3(coord-dim)]
        """
        dire_vect = self.forward(nir)  # [N, L, L, D]
        loss_vect = self.cri(dire_vect, abs_gap)
        
        path_vectors = self.find_all_paths_batched(dire_vect)  # [N, L, L, num_paths, D]
        
        # 计算所有路径与对应GT的L2距离
        loss_loop = torch.linalg.vector_norm(
            path_vectors - abs_gap.unsqueeze(3), 
            dim=-1
        ).mean()
        
        return {"loss_vect": loss_vect, "loss_loop": loss_loop}

    @torch.inference_mode()
    def predict(self, nir:Tensor, abs_gap: Tensor|None=None) -> tuple[Tensor, Tensor|None]:
        """
        Args:
            nir (Tensor): [N, num_views, C, Z, Y, X]
            
        Returns:
            vec_pred (Tensor): [N, num_views, num_views, 3] 
        """
        pred = self.forward(nir)
        if abs_gap is not None:
            loss = self.cri(pred, abs_gap)
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
                - abs_gap_gt (Absolute Gap): [num_views, num_views, 3 (coord-dims)]
                - rel_gap_gt (Relative Gap): [num_views, num_views, 3 (coord-dims)]
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
            "gap_loss": c("gap_loss", results),
            "sim_loss": c("sim_loss", results),
            "vec_loss": c("vec_loss", results),
        }
        print_log(context, logger='current')
        return context


class RelSim_VisHook(Hook):
    def __init__(self, interval:int, *args, **kwargs):
        self.interval = interval
        self._visualizer: RelSim_Viser = Visualizer.get_current_instance()
        super().__init__(*args, **kwargs)
    
    def after_val_iter(self,
                       runner:Runner,
                       batch_idx: int,
                       data_batch: dict|tuple|list|None = None,
                       outputs: Sequence|None = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        if outputs is not None and (batch_idx % self.interval == 0 or batch_idx == 0):
            self._visualizer.add_datasample(outputs[0], batch_idx)


class RelSim_Viser(Visualizer):
    def _plt2array(self, fig: plt.Figure) -> np.ndarray:
        fig.canvas.draw()
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
    
    def _vis_gap(self, gap: torch.Tensor, rel_gap: torch.Tensor):
        """
        Args:
            gap (Tensor): [num_views, num_views]
            rel_gap (Tensor): [num_views, num_views, coord_dims]
        """
        rel_gap = rel_gap.norm(dim=-1)  # [num_views, num_views]
        
        # 获取上三角矩阵的掩码
        mask = torch.triu(torch.ones_like(gap), diagonal=1).bool()
        
        # 提取上三角矩阵的值
        gap_values = gap[mask].cpu().numpy()
        rel_gap_values = rel_gap[mask].cpu().numpy()
        
        # 创建散点图
        plt.figure(figsize=(2, 2))
        plt.plot([0, 1], [0, 1], color=CMAP_COLOR[1], linestyle='--', zorder=0, alpha=0.7)
        plt.scatter(gap_values, rel_gap_values, color=CMAP_COLOR[0], alpha=0.6)
        
        # 设置图表属性
        plt.xlabel('Gap Prediction')
        plt.ylabel('Ground Truth')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xticks([0, 0.5, 1])
        plt.yticks([0, 0.5, 1])
        
        # to ndarray
        img_arr = self._plt2array(plt.gcf())
        plt.close()
        return img_arr
    
    def _vis_sim(self, 
                entire_volume: np.ndarray, 
                sub_views: torch.Tensor, 
                sub_view_coords: torch.Tensor, 
                adja_coords: torch.Tensor, 
                dist_coords: torch.Tensor, 
                sim_chunk_size: torch.Tensor):
        """
        A simple illustrative 3D example using matplotlib's 3D projection.
        It draws wireframes to approximate bounding boxes.
        Args:
            entire_volume (Tensor): [C, D, H, W]
            sub_views (Tensor): [num_views, C, subD, subH, subW]
            sub_view_coords (Tensor): [num_views, 3]
            adja_coords (Tensor): [num_pairs, 2, 3]
            dist_coords (Tensor): [num_pairs, 2, 3]
            sim_chunk_size (Tensor): [3 (coord_dims)]
        """
        
        pdb.set_trace()
        sub_views = sub_views.cpu()
        sub_view_coords = sub_view_coords.cpu()
        adja_coords = adja_coords.cpu()
        dist_coords = dist_coords.cpu()
        
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

        # 绘制所有sub_view
        _, _, subD, subH, subW = sub_views.shape
        for coord in sub_view_coords:
            draw_bbox(coord, 
                    (subD, subH, subW), 
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
                draw_bbox(ac, sim_chunk_size, color=DEFAULT_CMAP(32), alpha=0.9)
            # 两个dist
            for dc in dist_centers:
                draw_bbox(dc, sim_chunk_size, color=DEFAULT_CMAP(224), alpha=0.9)
        
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
            front_slice = entire_volume[0, :, :, -1]  # y-z平面 (x最大)
            top_slice = entire_volume[0, -1, :, :]    # x-y平面 (z最大)
            right_slice = entire_volume[0, :, -1, :]  # x-z平面 (y最大)
            
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

    def _vis_vec(self, vec:Tensor, coords:Tensor, abs_gap:Tensor):
        """
        Args:
            coords (Tensor): [sub-view, 3 (coord-dim)]
            vec (Tensor): [num_views, num_views, 3]
            abs_gap (Tensor): [num_views, num_views, 3]
        """
        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4,1])
        n = coords.shape[0]
        colors = [DEFAULT_CMAP(i/(n-1)) for i in range(n)]

        # 主图坐标
        ax_main = fig.add_subplot(gs[0,0], projection='3d')
        ax_main.scatter(coords[:,0], coords[:,1], coords[:,2], 
                    marker='x', 
                    c=colors)  # 使用colormap

        # 主图向量
        for i in range(n-1):
            source = coords[i]
            pred = vec[i, i+1]
            ax_main.quiver(*source, *pred, 
                        color=colors[i+1],  # 使用目标点的颜色
                        alpha=0.8, 
                        length=1, 
                        arrow_length_ratio=0.1)

        # 右侧三张子图
        def draw_subfig(gs: SubplotSpec, vec: Tensor, abs_gap: Tensor):
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
                gt = abs_gap[triu_indices[0], triu_indices[1], dim].cpu().numpy()
                
                # 散点图
                ax.scatter(pred, gt, alpha=0.5, s=20, color=CMAP_COLOR[1])
                # 对角线
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0)

                ax.set_title(f'{dim_names[dim]}')
                ax.set_aspect('equal')
        
        # 绘制子图
        draw_subfig(gs[0,1], vec, abs_gap)

        fig.tight_layout()
        img_arr = self._plt2array(fig)
        plt.close(fig)
        return img_arr

    @master_only
    def add_datasample(self, data_sample:dict, step:int|None=None):
        """
        Args:
            data_sample: datasample dict
                - volume: [C, ...]
                - sub_views: [num_views, C, ...]
                - coords: [num_views, 3]
                - abs_gap_gt (Absolute Gap): [num_views, num_views, 3 (coord-dims)]
                - rel_gap_gt (Relative Gap): [num_views, num_views, 3 (coord-dims)]
                - nir (Neural Implicit Representation): [num_views, C, ...]

                - gap_pred (Gap Prediction): [num_views, num_views]
                - sim_pred_coords (Similarity): [num_pairs, 4 (i_adj1, i_adj2, i_dist1, i_dist2), 3]
                - vec_pred (Vector): [num_views, num_views, 3]
                - sim_chunk_size (Tensor): [num_pairs, 3]
        """
        gap_vis_img = self._vis_gap(data_sample.gap_pred, 
                                    data_sample.rel_gap_gt)
        sim_vis_img = self._vis_sim(data_sample.volume, 
                                    data_sample.sub_views, 
                                    data_sample.coords, 
                                    data_sample.sim_pred[:, :2], 
                                    data_sample.sim_pred[:, 2:], 
                                    data_sample.sim_chunk_size)
        vec_vis_img = self._vis_vec(data_sample.vec_pred,
                                    data_sample.coords,
                                    data_sample.abs_gap_gt)
        self.add_image('Gap Prediction', gap_vis_img, step)
        self.add_image('Similarity Prediction', sim_vis_img, step)
        self.add_image('Vector Prediction', vec_vis_img, step)






"""MGAM TEST"""

def generate_test_data(batch_size:int=2, 
                      num_views:int=3,
                      channels:int=4,
                      volume_size:int=64) -> tuple[torch.Tensor, torch.Tensor]:
    """生成测试数据
    
    Returns:
        nir: [N, num_views, C, Z, Y, X]
        abs_gap: [N, num_views, num_views, 3]
    """
    # 生成随机体积数据
    nir = torch.randn(batch_size, num_views, channels, 
                     volume_size, volume_size, volume_size)
    
    # 生成随机距离向量
    abs_gap = torch.randn(batch_size, num_views, num_views, 3)
    
    return nir, abs_gap

class TestSimPairDiscriminator:
    @pytest.fixture
    def discriminator(self):
        return SimPairDiscriminator(
            sub_volume_size=48,
            dim="3d",
            in_channels=4
        )
    
    def test_initialization(self, discriminator):
        assert discriminator.s == 48
        assert discriminator.dim == "3d"
        assert discriminator.in_channels == 4
        assert len(discriminator.pair_encoder) == 4
    
    def test_forward(self, discriminator):
        batch_size, num_pairs = 2, 3
        num_sub_vols, channels = 4, 4
        vol_size = 48
        
        # 创建输入tensor
        sub_vols = torch.randn(batch_size, num_pairs, num_sub_vols, 
                             channels, vol_size, vol_size, vol_size)
        
        # 执行forward
        output = discriminator.forward(sub_vols)
        
        # 检查输出维度
        assert output.shape == (batch_size, num_pairs, 2)
    
    def test_loss_calculation(self, discriminator):
        # 生成测试数据
        nir, abs_gap = generate_test_data()
        
        # 计算损失
        loss = discriminator.loss(nir, abs_gap)
        
        # 检查损失是否为标量
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_sub_volume_selector(self, discriminator):
        # 生成测试数据
        nir, abs_gap = generate_test_data()
        
        # 获取子体积索引
        indices = discriminator._get_subvolume_indices(abs_gap, [nir.shape[0], *nir.shape[3:]])
        
        # 选择子体积
        sub_vols = discriminator._sub_volume_selector(nir, indices)
        
        # 检查输出维度
        assert len(sub_vols.shape) == 7  # [N, num_pairs, 4, C, s, s, s]
        assert sub_vols.shape[2] == 4    # 4个子体积
        assert all(s == 48 for s in sub_vols.shape[-3:])  # 子体积大小
    
    def test_generate_target(self, discriminator):
        batch_size, num_pairs = 2, 3
        dist_preds = torch.randn(batch_size, num_pairs, 2)
        
        target = discriminator._generate_target(dist_preds)
        
        assert target.shape == (batch_size, num_pairs, 2)
        assert torch.all(target[..., 0] == discriminator.LABEL_ADJA_PAIR)
        assert torch.all(target[..., 1] == discriminator.LABEL_DIST_PAIR)

def test_gap_predictor():
    # 测试3D情况
    batch_size = 2
    num_views = 3
    channels = 16
    spatial_size = 64
    
    predictor = GapPredictor(
        dim="3d",
        in_channels=channels,
        num_views=num_views
    )
    
    # 创建模拟输入
    nir = torch.randn(
        batch_size, 
        num_views,
        channels,
        spatial_size,
        spatial_size,
        spatial_size
    )
    
    # 测试forward方法
    similarity = predictor.forward(nir)
    
    # 验证输出形状
    assert similarity.shape == (batch_size, num_views, num_views)
    
    # 测试loss方法
    abs_gap = torch.randn(batch_size, num_views, num_views)
    loss = predictor.loss(nir, abs_gap)
    assert isinstance(loss, Tensor)
    assert loss.ndim == 0  # 标量损失值

def test_gap_predictor_different_dims():
    dims = ["1d", "2d", "3d"]
    spatial_sizes = {
        "1d": (64,),
        "2d": (64, 64),
        "3d": (64, 64, 64)
    }
    
    for dim in dims:
        batch_size = 2
        num_views = 3
        channels = 16
        
        predictor = GapPredictor(
            dim=dim,
            in_channels=channels,
            num_views=num_views
        )
        
        # 创建对应维度的输入
        input_shape = (batch_size, num_views, channels) + spatial_sizes[dim]
        nir = torch.randn(*input_shape)
        
        # 测试forward
        similarity = predictor.forward(nir)
        assert similarity.shape == (batch_size, num_views, num_views)

def test_gap_predictor_edge_cases():
    # 测试单batch情况
    predictor = GapPredictor("3d", 16, 3)
    nir = torch.randn(1, 3, 16, 48, 48, 48)
    similarity = predictor.forward(nir)
    assert similarity.shape == (1, 3, 3)
    
    # 测试最小volume数
    predictor = GapPredictor("3d", 16, 2)
    nir = torch.randn(2, 2, 16, 48, 48, 48)
    similarity = predictor.forward(nir)
    assert similarity.shape == (2, 2, 2)



if __name__ == "__main__":
    pytest.main([__file__])