from curses import window
import pdb

import torch
from torch import Tensor

from mmengine.logging import print_log, MMLogger
from mmengine.model.base_module import BaseModule
from mmseg.registry import MODELS

from mgamdata.mm.mmseg_Dev3D import EncoderDecoder_3D, PixelShuffle3D, PixelUnshuffle3D



class DynamicParam(BaseModule):
    def __init__(self, in_channels:int, mid_channels:int, dim:str, *args, **kwargs):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.normal_(self.param, mean=1, std=0.5)
    
    def forward(self, *args, **kwargs) -> Tensor:
        return self.param
    
    def __repr__(self):
        return "{:.3f}".format(self.param.item())


class WindowExtractor(BaseModule):
    "Extract values from raw array using a learnable window."
    
    def __init__(self,
                 in_channels:int,
                 embed_dims:int,
                 value_range:list[int],
                 eps:float=1.0,
                 log_interval:int|None=None,
                 dim='3d',
                 *args, **kwargs):
        assert len(value_range) == 2 and value_range[1] > value_range[0]
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.value_range = value_range
        self.eps = eps
        self.dim = dim
        self.log_interval = log_interval
        self.log_count = 0
        self.window_sample = torch.arange(*self.value_range)
        
        # Dynamic Weak Response - a
        self.d_wr = DynamicParam(in_channels, embed_dims, dim)
        # Dynamic Intense Response - b
        self.d_ir = DynamicParam(in_channels, embed_dims, dim)
        # Dynamic Perception Field - d
        self.d_pf = DynamicParam(in_channels, embed_dims, dim)
        # Dynamic Range - g
        self.d_r = DynamicParam(in_channels, embed_dims, dim)
        # Global Response - c
        self.g_r = DynamicParam(in_channels, embed_dims, dim)
        # Global Offset - k
        self.g_o = torch.nn.Parameter(torch.zeros(1))
        # Range Rectification Coefficient - h
        self.rrc = value_range[1] - value_range[0]
    
    @torch.inference_mode()
    def current_response(self):
        self.window_sample = self.window_sample.to(device=self.g_o.device, non_blocking=True)
        response = self.forward(self.window_sample, log_override=False).cpu().numpy()
        return response
    
    @torch.inference_mode()
    def log(self) -> str|None:
        if self.log_interval is not None:
            self.log_count = (self.log_count + 1) % self.log_interval
            if self.log_count == 0:
                sample_response = self.current_response()
                msg = (
                    f"d_wr: {repr(self.d_wr)}, "
                    f"d_ir: {repr(self.d_ir)}, "
                    f"d_pf: {repr(self.d_pf)}, "
                    f"d_r: {repr(self.d_r)}, "
                    f"g_r: {repr(self.g_r)}, "
                    f"g_o: {self.g_o.item():.3f}, "
                    f"rrc: {repr(self.rrc)}. "
                    f"std {sample_response.std():.2f}, "
                    f"max {sample_response.max():.2f}, "
                    f"min {sample_response.min():.2f}.")
                print_log(msg, MMLogger.get_current_instance())
                return msg
    
    def forward(self, x:Tensor, log_override:bool=True):
        """
        Args:
            inputs (Tensor): (...)
        """
        d_pf = self.d_pf(x)
        g_r = self.g_r(x)
        d_r = self.d_r(x)
        d_wr = self.d_wr(x)
        d_ir = self.d_ir(x)
        
        exp = (x / self.rrc + d_pf) \
            / (g_r * d_r)
        
        response = \
            d_r * (
                  d_wr * torch.exp( exp) \
                - d_ir * torch.exp(-exp)
            ) / (
                  d_wr * torch.exp( exp) \
                + d_ir * torch.exp(-exp)
            ) + (
            d_r * self.g_o
        )
        
        # Avoid infinite recursion during current_response calculation.
        if log_override:
            self.log()
        return response


class ValueWiseProjector(BaseModule):
    """
    Value-Wise Projector for one window remapping operation.
    The extracted value are fine-tuned by this projector.
    """

    def __init__(self, 
                 in_channels: int,
                 num_rect: int,
                 momentum: float=0.1,
                 valid_range:list[int] = [-1024, 3072],
                 dim:str='3d',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_rect = num_rect
        self.valid_range = valid_range
        self.dim = dim

        self.rectify_intense = torch.nn.Parameter(
            torch.ones(num_rect))
        self.rectify_location = torch.nn.Parameter(
            torch.zeros(num_rect))
        
        self.regulation_memory = []
        self.regulation_memory_limit = int(1 / (1 - momentum))

    @property
    @torch.inference_mode()
    def current_projection(self):
        sample_data = torch.arange(self.regulation_nbins).to(
            device=self.projection_coefficient.device)
        return self.forward(sample_data).cpu().numpy()

    def regulation(self, inputs:Tensor) -> Tensor:
        """
        Limit the projector ability to ensure it's behavior,
        which aligns with the physical meaning.
        """
        with torch.no_grad():
            iter_std = inputs.std()
            if not torch.isnan(iter_std):
                self.regulation_memory.append(inputs.std())
            self.regulation_memory = self.regulation_memory[-self.regulation_memory_limit:]
            target_std = torch.stack(self.regulation_memory).mean()
        
        regulation_loss = torch.abs(self.rectify_location.std() - target_std)
        return regulation_loss * len(self.regulation_memory) / self.regulation_memory_limit

    def forward(self, inputs:Tensor) -> Tensor|tuple[Tensor, Tensor]:
        """
        Args:
            inputs (Tensor): (...)
        """
        # TODO Tanh? Maybe improper here!
        rectified = torch.tanh(
            self.rectify_location   # [num_rect]
            + inputs.expand(self.num_rect, *inputs.shape).moveaxis(0,-1) # [..., num_rect]
        ) * self.rectify_intense # rectified: [..., num_rect]
        
        return torch.sum(rectified, dim=-1) # [...]


class BatchCrossWindowFusion(BaseModule):
    def __init__(self, num_windows:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_fusion_weight = torch.nn.Parameter(
            torch.ones(num_windows, num_windows))
    
    @property
    def current_fusion(self):
        return self.window_fusion_weight.detach().cpu().numpy()
    
    def forward(self, inputs:Tensor):
        """
        Args:
            inputs (Tensor): (Win, N, C, ...)
        
        Returns:
            Tensor: (N, Win*C, ...)
        """
        ori_shape = inputs.shape
        fused = torch.matmul(
            self.window_fusion_weight, 
            inputs.reshape(ori_shape[0], -1)
        ).reshape(ori_shape)
        
        # Window Concatenate
        window_concat_on_channel = fused.transpose(0,1).reshape(
            ori_shape[1], ori_shape[0]*ori_shape[2], *ori_shape[3:])
        
        return window_concat_on_channel # [N, Win*C, ...]


class ParalleledMultiWindowProcessing(BaseModule):
    """The top module of Paralleled Multi-Window Processing."""
    
    def __init__(self,
                 in_channels:int,
                 embed_dims:int,
                 window_embed_dims:int|None=None,
                 num_windows:int=4,
                 num_rect:int=8,
                 rect_momentum:float=0.99,
                 data_range:list[int]=[-1024, 3072],
                 dim='3d',
                 log_interval:int|None=None,
                 enable_VWP:bool=True,
                 *args, **kwargs
                ):
        assert dim.lower() in ['2d', '3d']
        super().__init__(*args, **kwargs)
        
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_windows = num_windows
        self.window_embed_dims = window_embed_dims \
                                 if window_embed_dims is not None else \
                                 embed_dims // self.num_windows
        self.num_rect = num_rect
        self.rect_momentum = rect_momentum
        self.data_range = data_range
        self.dim = dim
        self.log_interval = log_interval
        self.enable_VWP = enable_VWP
        self._init_PMWP()
    
    def _init_PMWP(self):
        for i in range(self.num_windows):
            setattr(self, f"window_extractor_{i}", 
                WindowExtractor(
                    in_channels=self.in_channels,
                    embed_dims=self.embed_dims,
                    value_range=self.data_range,
                    log_interval=self.log_interval,
                    dim=self.dim))
            setattr(self, f"value_wise_projector_{i}", 
                ValueWiseProjector(
                    in_channels=self.in_channels,
                    num_rect=self.num_rect,
                    momentum=self.rect_momentum,
                    valid_range=self.data_range,
                    dim=self.dim))
        
        # TODO Maybe Point-Wise Attention?
        self.cross_window_fusion = BatchCrossWindowFusion(self.num_windows)

    def forward(self, inputs:Tensor, regulation_weight:float=0.):
        """
        Args:
            inputs (Tensor): (N, C, ...)
        """
        C = inputs.size(1)
        x = []
        projector_aux_losses = []
        
        for i in range(self.num_windows):
            proj = getattr(self, f"window_extractor_{i}")(inputs)
            proj = getattr(self, f"value_wise_projector_{i}").forward(proj)
            x.append(proj)
            
            if regulation_weight != 0:
                projector_aux_loss = regulation_weight * getattr(
                    self, f"value_wise_projector_{i}").regulation(proj)
                projector_aux_losses.append(projector_aux_loss)
        
        x = torch.stack(x) # [Win, N, C, ...]
        x = self.cross_window_fusion(x) # [N, Win*C, ...]
        
        if regulation_weight != 0:
            projector_aux_losses = torch.stack(projector_aux_losses, dim=0).mean()
        else:
            projector_aux_losses = None
            
        return x, projector_aux_losses # [N, C, ...]


class AutoWindowSetting(EncoderDecoder_3D):
    "Compatible Plugin for Auto Window Setting."
    
    def __init__(self, pmwp:dict, regulation_weight:float=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pmwp:ParalleledMultiWindowProcessing = MODELS.build(pmwp)
        self.regulation_weight = regulation_weight
    
    def extract_feat(self, inputs:Tensor, regulation_weight:float=0.):
        # inputs: [N, C, ...]
        # pmwp_out: [N, num_window * C, ...]
        # TODO Downsampling Channel?
        pmwp_out, projector_aux_loss = self.pmwp(inputs, regulation_weight)
        
        if regulation_weight != 0:
            return super().extract_feat(pmwp_out), projector_aux_loss
        else:
            return super().extract_feat(pmwp_out)
    
    def loss(self, inputs: Tensor, data_samples):
        if self.regulation_weight != 0:
            x, projector_loss = self.extract_feat(inputs, self.regulation_weight)
            losses = dict(loss_aux_projector=projector_loss)
        else:
            x = self.extract_feat(inputs)
            losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
