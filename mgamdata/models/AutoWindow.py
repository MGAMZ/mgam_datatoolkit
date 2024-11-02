import pdb
from typing_extensions import deprecated

import torch
from torch import Tensor

from mmengine.logging import print_log, MMLogger
from mmengine.model.base_module import BaseModule
from mmseg.registry import MODELS

from mgamdata.mm.mmseg_Dev3D import EncoderDecoder_3D



class WindowExtractor(BaseModule):
    "Extract values from raw array using a learnable window."
    
    def __init__(self, 
                 value_range:list[int],
                 eps:float=1.0,
                 log_interval:int|None=None,
                 *args, **kwargs):
        assert len(value_range) == 2 and value_range[1] > value_range[0]
        super().__init__(*args, **kwargs)
        self.value_range = value_range
        self.window_sample = torch.arange(*self.value_range)
        self.eps = eps
        self.log_interval = log_interval
        self.log_count = 0
        
        # Dynamic Weak Response - a
        self.d_wr = torch.nn.Parameter(torch.ones(1))
        # Dynamic Intense Response - b
        self.d_ir = torch.nn.Parameter(torch.ones(1))
        # Dynamic Perception Field - d
        self.d_pf = torch.nn.Parameter(torch.ones(1))
        # Dynamic Range - g
        self.d_r = torch.nn.Parameter(torch.ones(1))
        # Global Response - c
        self.g_r = torch.nn.Parameter(torch.ones(1))
        # Range Rectification Coefficient - h
        self.rrc = value_range[1] - value_range[0]
    
    @property
    @torch.inference_mode()
    def current_response(self):
        self.window_sample = self.window_sample.to(device=self.dynamic_range.device)
        response = self.forward(self.window_sample).cpu().numpy()
        return response
    
    def log(self):
        if self.log_interval is not None:
            self.log_count = (self.log_count + 1) % self.log_interval
            if self.log_count == 0:
                sample_response = self.current_response()
                print_log(f"WinExt: d_wr: {self.d_wr.item():.4f}, "
                          f"d_ir: {self.d_ir.item():.4f}, "
                          f"d_pf: {self.d_pf.item():.4f}, "
                          f"d_r: {self.d_r.item():.4f}, "
                          f"g_r: {self.g_r.item():.4f}, "
                          f"rrc: {self.rrc}. "
                          f"std {sample_response.std():.2f}, "
                          f"max {sample_response.max():.2f}, "
                          f"min {sample_response.min():.2f}.")
    
    def forward(self, x:Tensor):
        """
        Args:
            inputs (Tensor): (...)
        """
        
        exp = (x / self.rrc + self.d_pf) \
            / (self.g_r * self.d_r)
        
        response = \
            self.d_r * (
                  self.d_wr * torch.exp( exp) \
                - self.d_ir * torch.exp(-exp)
            ) / (
                  self.d_wr * torch.exp( exp) \
                + self.d_ir * torch.exp(-exp)
            )
        
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

        if dim.lower() == '2d':
            self.pmwm_norm = torch.nn.InstanceNorm2d(
                num_features=in_channels,
                affine=True, 
                track_running_stats=True)
        else:
            self.pmwm_norm = torch.nn.InstanceNorm3d(
                num_features=in_channels,
                affine=True,
                track_running_stats=True)

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
        normed:Tensor = self.pmwm_norm(inputs) # [...]

        rectified = torch.tanh(
            self.rectify_location   # [num_rect]
            + normed.expand(
                self.num_rect, 
                *normed.shape
            ).moveaxis(0,-1) # [..., num_rect]
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
                 window_embed_dims:int=32,
                 window_width:int=200,
                 num_windows:int=4,
                 num_rect:int=8,
                 rect_momentum:float=0.1,
                 data_range:list[int]=[-1024, 3072],
                 dim='3d',
                 log_interval:int|None=None,
                 *args, **kwargs
                ):
        assert dim.lower() in ['2d', '3d']
        super().__init__(*args, **kwargs)
        
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.window_embed_dims = window_embed_dims
        self.window_width = window_width
        self.num_windows = num_windows
        self.num_rect = num_rect
        self.rect_momentum = rect_momentum
        self.data_range = data_range
        self.dim = dim
        self.log_interval = log_interval
        self._init_PMWP()
    
    def _init_PMWP(self):
        for i in range(self.num_windows):
            setattr(self, f"window_extractor_{i}", 
                WindowExtractor(
                    value_range=self.data_range,
                    log_interval=self.log_interval))
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
        x = []
        projector_aux_losses = []
        
        for i in range(self.num_windows):
            extracted = getattr(self, f"window_extractor_{i}")(inputs)
            # projected = getattr(self, f"value_wise_projector_{i}").forward(extracted)
            x.append(extracted)
            
            if regulation_weight != 0:
                projector_aux_loss = regulation_weight * getattr(
                    self, f"value_wise_projector_{i}").regulation(extracted)
                projector_aux_losses.append(projector_aux_loss)
        
        x = torch.stack(x, dim=0) # [W, N, C, ...]
        x = self.cross_window_fusion(x) # [N, Win*C, ...]
        
        if regulation_weight != 0:
            projector_aux_losses = torch.stack(projector_aux_losses, dim=0).mean()
        else:
            projector_aux_losses = None
            
        return x, projector_aux_losses # [N, Win*C, ...]


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
