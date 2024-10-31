import pdb

import torch
from torch import Tensor

from mmengine.model.base_module import BaseModule
from mmseg.registry import MODELS

from mgamdata.mm.mmseg_Dev3D import EncoderDecoder_3D



class WindowExtractor(BaseModule):
    "Extract values from raw array using a learnable window."

    DYNAMIC_WEAK_RESPONSE_AMPLIFIER = 10
    DYNAMIC_INTENSE_RESPONSE_AMPLIFIER = 10
    DYNAMIC_RANGE_AMPLIFIER = 100

    def __init__(self, 
                 value_range:list[int],
                 eps:float=1e-6,
                 *args, **kwargs):
        assert len(value_range) == 2 and value_range[1] > value_range[0]
        super().__init__(*args, **kwargs)
        self.value_range = value_range
        self.window_sample = torch.arange(*self.value_range)
        self.dynamic_range = torch.nn.Parameter(
            torch.tensor([0.5], dtype=torch.float32))
        self.dynamic_weak_response = torch.nn.Parameter(
            torch.tensor([1], dtype=torch.float32))
        self.dynamic_intense_response = torch.nn.Parameter(
            torch.tensor([1], dtype=torch.float32))
        self.eps = eps
    
    @property
    @torch.inference_mode()
    def current_response(self):
        self.window_sample = self.window_sample.to(device=self.dynamic_range.device)
        response = self.forward(self.window_sample).cpu().numpy()
        return response
    
    
    def forward(self, x:Tensor):
        """
        :math:: response = 4096\frac{1\exp\left(\frac{x}{4096}\right)-400\exp\left(-\frac{x}{4096}\right)}{1\exp\left(\frac{x}{4096}\right)+400\exp\left(-\frac{x}{4096}\right)}
        
        Args:
            inputs (Tensor): (...)
        """
        d_r = self.dynamic_range * self.DYNAMIC_RANGE_AMPLIFIER
        d_wr = self.dynamic_weak_response * self.DYNAMIC_WEAK_RESPONSE_AMPLIFIER
        d_ir = self.dynamic_intense_response * self.DYNAMIC_INTENSE_RESPONSE_AMPLIFIER
        d_r = torch.relu(d_r) + self.eps
        d_wr = torch.relu(d_wr) + self.eps
        d_ir = torch.relu(d_ir) + self.eps
        
        response = \
            d_r * (
                  d_wr * torch.exp( x / d_r) \
                - d_ir * torch.exp(-x / d_r)
            ) / (
                  d_wr * torch.exp( x / d_r) \
                + d_ir * torch.exp(-x / d_r)
            )
        
        return response



class ValueWiseProjector(BaseModule):
    """
    Value-Wise Projector for one window remapping operation.
    The extracted value are fine-tuned by this projector.
    """
    
    def __init__(self, 
                 in_channels:int, 
                 order:int=1,
                 regulation_nbins:int=512,
                 valid_range:list[int] = [-1024, 3072],
                 dim:str='3d',
                 *args, **kwargs):
        assert order > 0, f"The order of projector should be greater than 0, got {order}."
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.order = order
        self.regulation_nbins = regulation_nbins
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
        
        self.projection_coefficient = torch.nn.Parameter(
            torch.cat([torch.ones(1), torch.zeros(order-1)]))
        self.projection_bias = torch.nn.Parameter(torch.randn(1) / 10)
        self.projection_exponent = torch.arange(1, order+1)
        
        self.sample_data = torch.arange(
            *self.valid_range, 
            step=self.regulation_nbins,
            pin_memory=True
            )[None,None,None].to(
                device=self.projection_coefficient.device,
                dtype=torch.float32)
    
    @property
    @torch.inference_mode()
    def current_projection(self):
        sample_data = torch.arange(self.regulation_nbins).to(
            device=self.projection_coefficient.device)
        return self.forward(sample_data).cpu().numpy()
    
    
    def regulation(self):
        """
        Limit the projector ability to ensure it's behavior,
        which aligns with the physical meaning.
        """
        self.sample_data = self.sample_data.to(device=self.projection_bias.device)
        projected_value = self.forward(self.sample_data)
        ascend_regulation = (projected_value.diff() - 1).abs().mean()
        smoothness_regulation = (projected_value.diff() - 1).std()
        return ascend_regulation + smoothness_regulation
    
    
    def forward(self, inputs:Tensor) -> Tensor:
        """
        Args:
            inputs (Tensor): (N, C, ...)
        """
        
        response:torch.Tensor = self.pmwm_norm(inputs)
        
        """
        high_order_mapping = 
            x^order * W_order + 
            x^(order-1) * W_(order-1) +
            ...
            x^4 * W_4 (projection_coefficient[4]) + 
            x^3 * W_3 (projection_coefficient[3]) + 
            x^2 * W_2 (projection_coefficient[2]) + 
            x^1 * W_1 (projection_coefficient[1]) + 
            W_0 (projection_bias)
        
        Args:
            inputs (Tensor): (...)
        """
        self.projection_exponent = self.projection_exponent.to(device=inputs.device)
        projected = response.expand(self.order, *response.shape).moveaxis(0, -1) # [..., order]
        projected = torch.pow(projected, self.projection_exponent) # [..., order]
        projected = torch.matmul(projected, self.projection_coefficient) # [...]
        projected += self.projection_bias # [...]
        return projected # [...]



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
                 num_bins:int=512,
                 proj_order:int=1,
                 data_range:list[int]=[-1024, 3072],
                 dim='3d',
                 *args, **kwargs
                ):
        assert dim.lower() in ['2d', '3d']
        super().__init__(*args, **kwargs)
        
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.window_embed_dims = window_embed_dims
        self.window_width = window_width
        self.num_windows = num_windows
        self.num_bins = num_bins
        self.proj_order = proj_order
        self.data_range = data_range
        self.dim = dim
        self._init_PMWP()
    

    def _init_PMWP(self):
        for i in range(self.num_windows):
            setattr(self, f"window_extractor_{i}", 
                WindowExtractor(
                    value_range=self.data_range))
            setattr(self, f"value_wise_projector_{i}", 
                ValueWiseProjector(
                    in_channels=self.in_channels,
                    order=self.proj_order,
                    regulation_nbins=self.num_bins,
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
            projected = getattr(self, f"value_wise_projector_{i}").forward(extracted)
            x.append(projected)
            
            if regulation_weight != 0:
                projector_aux_loss = regulation_weight * getattr(
                    self, f"value_wise_projector_{i}").regulation()
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
