import os
import pdb
import pytest
from abc import abstractmethod
from functools import partial
from typing_extensions import Literal, OrderedDict
from itertools import product

import torch
from torch import Tensor
from torch.nn import PixelUnshuffle as PixelUnshuffle2D

from mmengine.model import MomentumAnnealingEMA
from mmengine.utils.misc import is_list_of
from mmengine.model.base_module import BaseModule
from mmengine.dist import all_gather, get_rank, is_main_process
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.models.selfsup.base import BaseSelfSupervisor
from mmpretrain.models.selfsup.mocov3 import CosineEMA

from mgamdata.mm.mmseg_Dev3D import PixelUnshuffle1D, PixelUnshuffle3D



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


class RelativeSimilaritySelfSup(AutoEncoderSelfSup):
    def __init__(self, momentum=1e-4, gamma=100, update_interval=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum_encoder = MomentumAnnealingEMA(
            self.whole_model_,
            momentum=momentum,
            gamma=gamma,
            interval=update_interval,
        )

    def parse_target(self, data_samples: list[DataSample]):
        # coords (coordinates):        [N, 3 (sub-volume), 3 (coord-dim)]
        coords = torch.stack([sample.coord for sample in data_samples])
        # abs_gap (absolute distance): [N, 3 (start from), 3 (point to), 3(coord-dim)]
        abs_gap = torch.stack([sample.abs_gap for sample in data_samples])
        # rel_gap (relative distance): [N, 3 (start from), 3 (point to), 3(coord-dim)]
        rel_gap = abs_gap / torch.triu(abs_gap, diagonal=1).sum(dim=(1,2))
        return coords, abs_gap, rel_gap

    def loss(
        self, inputs: list[Tensor], data_samples: list[DataSample], **kwargs
    ) -> dict[str, Tensor]:
        
        assert isinstance(inputs, list)
        self.backbone: BaseModule
        self.head: BaseModule
        
        # sv: sub volume
        sv1 = inputs[0]
        sv2 = inputs[1]
        sv3 = inputs[2]
        coords, abs_gap, rel_gap = self.parse_target(data_samples)
        
        # nir_sv: neural implicit representation of sub-volume
        nir_sv1 = self.whole_model_(sv1)[0]
        nir_sv2 = self.momentum_encoder(sv2)[0]
        nir_sv3 = self.momentum_encoder(sv3)[0]
        nir = torch.stack([nir_sv1, nir_sv2, nir_sv3], dim=1)  # [N, 3, C, Z, Y, X]
        
        # relative gap self-supervision
        rel_gap_loss = self.gap_head.loss(nir, abs_gap)
        # similarity self-supervision (Opposite nir has larger difference, namely negative label)
        nir_sim_loss = self.sim_head.loss(nir, coords)
        # vector angle self-supervision (Add Sum 180°)
        vec_ang_loss = self.vec_head.loss(nir, abs_gap)
        
        # update momentum model
        self.momentum_encoder.update_parameters(self.whole_model_)

        return {
            "loss_rel_gap": rel_gap_loss,
            "loss_nir_sim": nir_sim_loss,
            "loss_vec_ang": vec_ang_loss,
        }


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
    def __init__(self, dim:Literal["1d","2d","3d"], in_channels:int, num_volume:int=3):
        self.dim = dim
        self.num_volume = num_volume
        self.act = torch.nn.GELU()
        self.embed = eval(f"torch.nn.Conv{dim}")(in_channels, in_channels, 1)
        self.avg_pool = GlobalAvgPool(dim)
    
    def forward(self, x:Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, num_volume, C, Z, Y, X]
        
        Returns:
            x (Tensor): [N, num_volume, C]
        """
        num_volume = x.size(1)
        # forward each volume
        x = torch.stack(
            x.chunk(
                self.act(self.embed(num_volume)), 
                dim=1
            ), 
            dim=1
        )  # [N, num_volume, C, ...]
        x = self.avg_pool(x) # [N, num_volume, C, *1]
        return x  # [N, num_volume, C]


class GapPredictor(BaseVolumeWisePredictor):
    def __init__(self, dim:Literal["1d","2d","3d"], in_channels:int):
        super().__init__(dim, in_channels)
        self.proj = torch.nn.ModuleList([
            torch.nn.Linear(in_channels//(2**i), in_channels//(2**(i+1)))
            for i in range(4)
        ])
        self.act = torch.nn.GELU()
        self.cri = torch.nn.SmoothL1Loss()
    
    def forward(self, nir:Tensor) -> Tensor:
        """
        Args:
            nir (Tensor): Size [N, num_volume, C, Z, Y, X]
            rel_gap (Tensor): Size [N, num_volume (start from), num_volume (point to), coord-dim-length]
        
        Returns:
            vector gap sort loss (Tensor): [N, ]
        """
        nir = super().forward(nir)  # [N, num_volume, C]
        for proj in self.proj:
            nir = proj(nir)
            nir = self.act(nir)
        
        # Relative Positional Representation of Each Sub-Volume
        # The origin may align with the mean value of all samples' world coordinate systems'origin.
        # diff equals to the relative distance between each `nir`.
        rel_pos_rep_diff = nir.unsqueeze(2) - nir.unsqueeze(1)  # (N, num_volume, num_volume, C)
        # calculate the distance of `rel_pos_rep_diff`
        similarity = rel_pos_rep_diff.norm(dim=-1)  # (N, num_volume, num_volume)
        
        return similarity  # (N, num_volume, num_volume)
    
    def loss(self, nir:Tensor, abs_gap:Tensor) -> Tensor:
        # nir: []
        similarity = self.forward(nir)  # (N, num_volume, num_volume)
        loss = self.cri(similarity, abs_gap)
        return loss


class SimPairDiscriminator(BaseModule):
    LABEL_ADJA_PAIR = 0
    LABEL_DIST_PAIR = 1
    
    def __init__(self, sub_volume_size:int|list[int], dim:Literal["1d","2d","3d"], in_channels:int):
        super().__init__()
        self.s = sub_volume_size
        self.dim = dim
        self.in_channels = in_channels
        
        self.pair_encoder = torch.nn.ModuleList([
            eval(f"torch.nn.Conv{dim}")(2*in_channels*(2**i), 2*in_channels*(2**(i+1)), 3, 2)
            for i in range(4)
        ])
        self.act = torch.nn.GELU()
        self.avg_pool = GlobalAvgPool(dim)
        self.discriminator = torch.nn.Linear(2 * in_channels*(2**4), 1, bias=False)
        self.cri = torch.nn.BCELoss()
    
    def _get_subvolume_indices(self, diff_vectors, volume_shape
        ) -> list[
                  dict[
                       tuple[int,int], 
                       tuple[int, list[slice], list[slice]]
                      ]
                 ]:
        """
        Args:
            diff_vectors: shape [N, *extra_dims, d], where d = len(volume_shape) - 1
            volume_shape: (N, D1, D2, ..., Dd)

        Returns:
            a list of dictionaries where each element :
                indices[n][(i1, i2, ...)] = (n, [slice, slice, ..., slice])
                    i1 = (0,0), i2 = (0,1), ...
                n is batch index
        
        NOTE for Usage:
            (n, adjacent_slices, distant_slices) = indices[3][(2,4)]
            Volumes[n, :, *adjacent_slices] or Volumes[n, :, *distant_slices]
        
        NOTE
            The highest dimension of indices is batch, which is easy to understand, 
            as all batches remain independent.

            After selecting the batch, 
            the next step is to choose the source Volume and the target Volume, 
            and combine them into a Tuple. 
            For example, 
            if I want to obtain the indices of proximity and distance in Volume 1 relative to Volume 2, 
            I should choose the Tuple (1,2).

            After selection, 
            indices will return a Tuple containing three elements: 
            1) the first value represents the batch index, 
            2) the second value contains the indices of the subset in Volume 1 that is close to Volume 2, 
            3) the third value contains the indices of the subset in Volume 1 that is far from Volume 2.
        """
        
        N = volume_shape[0]
        dims = volume_shape[1:]  # D1, D2, ..., Dd
        shape_diff = diff_vectors.shape  # [N, *extra_dims, d]
        assert len(dims) == shape_diff[-1], f"The last dimension of diff_vectors ({shape_diff[-1]}) should be equal to the length of `volume_shape` - 1 ({len(dims)})"
        extra_dims = shape_diff[1:-1]  # exclude N and d, leaving only the extra dimensions

        indices = [dict() for _ in range(N)]

        for n in range(N):
            # iterate through all combinations of extra_dims
            for idx in product(*(range(s) for s in extra_dims)):
                # Extract the diff values along the last dimension, which correspond to the d coordinates
                diff = diff_vectors[(n,) + idx]  # shape [d]
                adjacent_slices = []
                distant_slices = []
                
                for dim_i in range(len(dims)):
                    size = dims[dim_i]
                    # Decide the slice based on the sign of diff[dim_i]
                    adjacent_slice_part = slice(None, self.s) \
                                 if diff[dim_i].item() < 0 \
                                 else slice(size-self.s, None)
                    distant_slice_part = slice(size-self.s, None) \
                                         if diff[dim_i].item() < 0 \
                                         else slice(None, self.s)
                    adjacent_slices.append(adjacent_slice_part)
                    distant_slices.append(distant_slice_part)
                
                indices[n][idx] = (n, adjacent_slices, distant_slices)

        return indices

    def _sub_volume_selector(self, nir:Tensor, sub_volume_indices):
        batched_samples = []
        for n in range(len(sub_volume_indices)):
            processed_pairs = set()
            samples = []
            for (vol_from, vol_to) in sub_volume_indices[n].keys():
                if (vol_from, vol_to) in processed_pairs or (vol_to, vol_from) in processed_pairs:
                    continue
                    
                # fetch index
                _, adj_from_to, dist_from_to = sub_volume_indices[n][(vol_from, vol_to)]
                _, adj_to_from, dist_to_from = sub_volume_indices[n][(vol_to, vol_from)]
                
                # select adjacent sub-nir
                v_from_adj = nir[n, vol_from, :, *adj_from_to]
                v_to_adj = nir[n, vol_to, :, *adj_to_from]
                
                # select distant sub-nir
                v_from_dist = nir[n, vol_from, :, *dist_from_to]
                v_to_dist = nir[n, vol_to, :, *dist_to_from]
                
                # encode, sample: [4, C, ...]
                sample = torch.stack([v_from_adj, v_to_adj, v_from_dist, v_to_dist], dim=0)
                samples.append(sample)
                processed_pairs.add((vol_from, vol_to))  # mark as processed
            
            batched_samples.append(torch.stack(samples, dim=0)) # [num_pairs, 4, C, ...]
            
        return torch.stack(batched_samples, dim=0)  # [N, num_pairs, 4, C, ...]

    def forward(self, sub_vols:Tensor) -> Tensor:
        """
        Args: 
            sub_vols (Tensor): [N, num_pairs, 4, C, ...]
            
        Returns:
            encoded_vols (Tensor): [N, num_pairs, 2 (adjacent, distant)]
        
        NOTE
            The third dimension 4 equals to [adj1, adj2, dist1, dist2]
        """
        
        ori_shape = sub_vols.shape
        # sub_vols: [N, num_pairs*2, 2C, ...]
        sub_vols = sub_vols.reshape(ori_shape[0], ori_shape[1]*2, 2*ori_shape[3], *ori_shape[4:])
        dist_preds = []
        
        for pair in range(sub_vols.size(1)):
            v = sub_vols[:, pair]  # [N, 2C, ...]
            for enc_layer in self.pair_encoder:
                v = enc_layer(v)
                v = self.act(v)
            v = self.avg_pool(v)  # [N, 2C]
            v = self.discriminator(v).squeeze(-1)  # [N, ]
            dist_preds.append(v)
        
        # dist_preds: [N, num_pairs, 2]
        dist_preds = torch.stack(dist_preds, dim=-1).reshape(ori_shape[0], ori_shape[1], 2)
        
        return dist_preds # [N, num_pairs, 2]

    def _generate_target(self, dist_preds:Tensor) -> Tensor:
        """Generate pseudo-labels for discriminator predictions
        
        Args:
            dist_preds (Tensor): shape [N, num_pairs, 2]
            
        Returns:
            target (Tensor): shape [N, num_pairs, 2]
        """
        assert dist_preds.size(-1) == 2, f"dist_preds should have 2 channels, but got {dist_preds.size(-1)}"
        if not hasattr(self, "pseudo_label"):
            target = torch.empty_like(dist_preds)
            target[..., 0] = self.LABEL_ADJA_PAIR  # adjacent pairs
            target[..., 1] = self.LABEL_DIST_PAIR  # distant pairs
            setattr(self, "pseudo_label", target)  # [N, num_pairs, 2]

        return getattr(self, "pseudo_label")  # [N, num_pairs, 2]

    def _binarize_preds(self, dist_preds: Tensor) -> Tensor:
        """Binarize predictions based on comparison along last dimension
        
        Args:
            dist_preds (Tensor): shape [N, num_pairs, 2]
            
        Returns:
            binary_preds (Tensor): shape [N, num_pairs, 2]
        """
        return torch.argsort(dist_preds, dim=-1).float()

    def loss(self, nir:Tensor, abs_gap:Tensor) -> Tensor:
        """
        Args:
            neural implicit representation (Tensor): [N, num_volume, C, Z, Y, X]
            abs_gap (Tensor): [N, num_volume (start from), num_volume (point to), coord-dim-length]
        """
        
        # determine the adjacent and distant sub-volumes' position
        sub_volume_indices = self._get_subvolume_indices(abs_gap, [nir.shape[0], *nir.shape[3:]])
        
        # get view of these positions
        sub_volumes = self._sub_volume_selector(nir, sub_volume_indices)  # [N, num_pairs, 4, C, ...]
        
        # execute prediction forward
        dist_preds = self.forward(sub_volumes)  # [N, num_pairs, 2 (adja, dist)]
        # binarize the prediction
        dist_preds = self._binarize_preds(dist_preds)  # [N, num_pairs, 2 (adja, dist)]
        
        # generate pseudo-label using adjacent and distant contexts
        target = self._generate_target(dist_preds)

        # execute loss calculation
        return self.cri(dist_preds, target)



"""MGAM TEST PASSED @ 2024.12.25"""


def generate_test_data(batch_size:int=2, 
                      num_volumes:int=3,
                      channels:int=4,
                      volume_size:int=64) -> tuple[torch.Tensor, torch.Tensor]:
    """生成测试数据
    
    Returns:
        nir: [N, num_volumes, C, Z, Y, X]
        abs_gap: [N, num_volumes, num_volumes, 3]
    """
    # 生成随机体积数据
    nir = torch.randn(batch_size, num_volumes, channels, 
                     volume_size, volume_size, volume_size)
    
    # 生成随机距离向量
    abs_gap = torch.randn(batch_size, num_volumes, num_volumes, 3)
    
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


if __name__ == "__main__":
    pytest.main([__file__])