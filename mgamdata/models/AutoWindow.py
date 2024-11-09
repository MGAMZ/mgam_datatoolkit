from collections.abc import Callable
import pdb
from io import BytesIO
from PIL import Image
from functools import partial

import seaborn
import numpy as np
import torch
from torch import Tensor
from matplotlib import pyplot as plt

from mmengine.runner.runner import Runner
from mmengine.hooks.hook import Hook
from mmengine.logging import print_log, MMLogger
from mmengine.model.base_module import BaseModule
from mmseg.registry import MODELS

from mgamdata.mm.mmseg_Dev3D import EncoderDecoder_3D



def scale_gradients(module, grad_input, grad_output):
    return tuple(g * module.lr_mult 
                 if g is not None else g
                 for g in grad_input)


class DynamicParam(BaseModule):
    def __init__(self, init_method:Callable, ensure_sign:str|None=None, eps:float=1e-2):
        super().__init__()
        if ensure_sign is not None:
            assert ensure_sign in ['pos', 'neg']
        self.ensure_sign = ensure_sign
        self.param = torch.nn.Parameter(torch.empty(1))
        init_method(self.param)
        self.eps = eps
    
    def forward(self, *args, **kwargs) -> Tensor:
        if self.ensure_sign == 'pos':
            return torch.relu(self.param) + self.eps
        elif self.ensure_sign == 'neg':
            return -torch.relu(self.param) - self.eps
        else:
            return self.param
    
    def status(self):
        return self.param.detach().cpu().numpy()
    
    def __repr__(self):
        return "{:.3f}".format(self.param.item())

    def __getattr__(self, name):
        if name == 'device':
            return self.param.device
        else:
            return super().__getattr__(name)


class SupportLrMultModule(BaseModule):
    def __init__(self, lr_mult:float|None=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_mult = lr_mult
        if lr_mult is not None:
            self.register_full_backward_hook(scale_gradients)


class WindowExtractor(SupportLrMultModule):
    "Extract values from raw array using a learnable window."
    
    def __init__(self,
                 in_channels:int,
                 embed_dims:int,
                 value_range:list|tuple,
                 lr_scale:float|None=None,
                 eps:float=1.0,
                 dim='3d',
                 *args, **kwargs):
        assert len(value_range) == 2 and value_range[1] > value_range[0]
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.value_range = value_range
        self.lr_scale = lr_scale
        self.eps = eps
        self.dim = dim
        self.log_count = 0
        self.window_sample = torch.arange(*self.value_range)
        
        # Dynamic Weak Response - a
        self.d_wr = DynamicParam(partial(torch.nn.init.normal_, mean=1, std=0.1), 
                                 ensure_sign='pos')
        # Dynamic Intense Response - b
        self.d_ir = DynamicParam(partial(torch.nn.init.normal_, mean=1, std=0.1), 
                                 ensure_sign='pos')
        # Dynamic Perception Field - d
        self.d_pf = DynamicParam(partial(torch.nn.init.normal_, mean=1, std=0.1), 
                                 ensure_sign=None)
        # Dynamic Range - g
        self.d_r = DynamicParam(partial(torch.nn.init.normal_, mean=1, std=0.1), 
                                ensure_sign='pos')
        # Global Response - c
        self.g_r = DynamicParam(partial(torch.nn.init.ones_), 
                                ensure_sign='pos')
        # Global Offset - k
        self.g_o = DynamicParam(partial(torch.nn.init.zeros_), 
                                ensure_sign=None)
        # Range Rectification Coefficient - h
        self.rrc = value_range[1] - value_range[0]
        # Major dynamic range for this window to handle.
        self.major_handle = (value_range[1] + value_range[0]) / 2
        assert self.rrc > 0
    
    @torch.inference_mode()
    def current_response(self):
        self.window_sample = self.window_sample.to(device=self.g_o.device, non_blocking=True)
        response = self.forward(self.window_sample).cpu().numpy()
        return response
    
    @torch.inference_mode()
    def status(self) -> dict:
        sample_response = self.current_response()
        return {
            "d_wr": self.d_wr.status(),
            "d_ir": self.d_ir.status(),
            "d_pf": self.d_pf.status(),
            "d_r": self.d_r.status(),
            "g_r": self.g_r.status(),
            "g_o": self.g_o.status(),
            "rrc": self.rrc,
            "RespStd": sample_response.std(),
            "RespMax": sample_response.max(),
            "RespMin": sample_response.min(),
            "RespMat": (self.window_sample.cpu().numpy(), sample_response)
        }
    
    def forward(self, x:Tensor):
        """
        Args:
            inputs (Tensor): (...)
        """
        d_pf = self.d_pf(x)
        g_r = self.g_r(x)
        d_r = self.d_r(x)
        d_wr = self.d_wr(x) + 1
        d_ir = self.d_ir(x) + 1
        g_o = self.g_o(x)
        
        exp = (
                (
                    x - self.major_handle
                ) / self.rrc 
                + d_pf
            ) / (
                g_r * d_r
            )
            
        response = \
            d_r * (
                  d_wr * torch.exp( exp) \
                - d_ir * torch.exp(-exp)
            ) / (
                  d_wr * torch.exp( exp) \
                + d_ir * torch.exp(-exp)
            ) + (
            d_r * g_o
        )
        
        return response


class ValueWiseProjector(SupportLrMultModule):
    """
    Value-Wise Projector for one window remapping operation.
    The extracted value are fine-tuned by this projector.
    """

    def __init__(self, 
                 in_channels: int,
                 num_rect: int,
                 momentum: float=0.1,
                 sample_nbins:int=256,
                 dim:str='3d',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_rect = num_rect
        self.sample = torch.arange(sample_nbins) / sample_nbins * 10 - 5
        self.dim = dim

        self.rectify_intense = torch.nn.Parameter(
            torch.randn(num_rect) * 0.1)
        self.rectify_location = torch.nn.Parameter(
            torch.randn(num_rect))
        
        self.momentum = momentum
        self.std_memory = torch.ones(1)

    @torch.inference_mode()
    def current_projection(self):
        self.sample = self.sample.to(device=self.rectify_intense.device)
        proj = self.forward(self.sample)
        return (self.sample.detach().cpu().numpy(), proj.cpu().numpy())

    def loss(self, inputs:Tensor) -> Tensor:
        """
        Limit the projector ability to ensure it's behavior,
        which aligns with the physical meaning.
        """
        with torch.no_grad():
            iter_std = inputs.std()
            if not torch.isnan(iter_std):
                self.std_memory = self.std_memory * self.momentum \
                                + iter_std * (1 - self.momentum)
        
        loss = torch.abs(self.rectify_location.std() - self.std_memory)
        return loss * len(self.std_memory) / self.std_memory_limit

    def forward(self, inputs:Tensor) -> Tensor:
        """
        Args:
            inputs (Tensor): (...)
        """
        # NOTE The accumulation operation is equivalent to the X-axis translation operation.
        rectification = torch.tanh(
            self.rectify_location   # [num_rect]
            + inputs.expand(self.num_rect, *inputs.shape).moveaxis(0,-1) # [..., num_rect]
        ) * self.rectify_intense # rectification: [..., num_rect]
        return inputs + torch.sum(rectification, dim=-1) # [...]


class BatchCrossWindowFusion(SupportLrMultModule):
    def __init__(self, num_windows:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_fusion_weight = torch.nn.Parameter(
            torch.eye(num_windows, num_windows))
    
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
            torch.softmax(self.window_fusion_weight, dim=1), 
            inputs.flatten(1)
        ).reshape(ori_shape)
        return fused.transpose(0,1).flatten(1,2) # [N, Win*C, ...]


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
                 enable_VWP:bool=True,
                 enable_CWF:bool=True,
                 lr_mult:float|None=None,
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
        self.enable_VWP = enable_VWP
        self.enable_CWF = enable_CWF
        self.lr_mult = lr_mult
        self._init_PMWP()
    
    def _init_PMWP(self):
        sub_window_ranges = self._split_range_into_milestones()
        
        for i in range(self.num_windows):
            setattr(self, f"window_extractor_{i}", 
                WindowExtractor(
                    in_channels=self.in_channels,
                    embed_dims=self.embed_dims,
                    value_range=sub_window_ranges[i],
                    dim=self.dim,
                    lr_mult=self.lr_mult))
            if self.enable_VWP:
                setattr(self, f"value_wise_projector_{i}", 
                    ValueWiseProjector(
                        in_channels=self.in_channels,
                        num_rect=self.num_rect,
                        momentum=self.rect_momentum,
                        dim=self.dim,
                        lr_mult=self.lr_mult))
        
        # TODO Maybe Point-Wise Attention?
        if self.enable_CWF:
            self.cross_window_fusion = BatchCrossWindowFusion(self.num_windows, 
                                                              lr_mult=self.lr_mult)

    def _split_range_into_milestones(self):
        if self.num_windows <= 0:
            raise ValueError("n must be a positive integer")
        start, end = self.data_range
        n = self.num_windows
        
        step = (end - start) / n
        milestones = np.arange(start, end + step, step)
        sub_ranges = [(milestones[i], milestones[i+1]) for i in range(n)]
        return sub_ranges

    def status(self):
        WinE = dict()
        for i in range(self.num_windows):
            for k, v in getattr(self, f"window_extractor_{i}").status().items():
                WinE[f"WinE/W{i}_{k}"] = v
        if self.enable_VWP:
            VluP = {
                f"VluP/P{i}": getattr(self, f"value_wise_projector_{i}").current_projection()
                for i in range(self.num_windows)
            }
        else: 
            VluP = None
        
        if self.enable_CWF:
            CrsF = self.cross_window_fusion.current_fusion
        else: 
            CrsF = None
        
        return WinE, VluP, CrsF

    def forward(self, inputs:Tensor, enable_pmwp_loss:bool=False):
        """
        Args:
            inputs (Tensor): (N, C, ...)
        """
        x = []
        projector_aux_losses = []
        
        for i in range(self.num_windows):
            proj = getattr(self, f"window_extractor_{i}")(inputs)
            if self.enable_VWP:
                proj = getattr(self, f"value_wise_projector_{i}").forward(proj)
            x.append(proj)
            
            if enable_pmwp_loss and self.enable_VWP:
                projector_aux_losses.append(getattr(
                    self, f"value_wise_projector_{i}").loss(proj))
        
        x = torch.stack(x) # [Window, N, C, ...]
        if self.enable_CWF:
            x = self.cross_window_fusion(x) # [N, Win*C, ...]
        else:
            x = x.transpose(0,1).flatten(1,2)
        
        pmwp_loss = {}
        if enable_pmwp_loss:
            projector_aux_losses = torch.stack(projector_aux_losses, dim=0).mean()
            pmwp_loss["loss_VluP"] = projector_aux_losses
        
        return x, pmwp_loss # [N, C, ...]


class AutoWindowSetting(EncoderDecoder_3D):
    "Compatible Plugin for Auto Window Setting."
    
    def __init__(self, 
                 pmwp:dict, 
                 enable_pmwp_loss:bool=False, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pmwp:ParalleledMultiWindowProcessing = MODELS.build(pmwp)
        self.enable_pmwp_loss = enable_pmwp_loss
    
    def extract_feat(self, inputs:Tensor, enable_pmwp_loss:bool=False):
        # inputs: [N, C, ...]
        # pmwp_out: [N, num_window * C, ...]
        # TODO Downsampling Channel?
        pmwp_out, pmwp_loss = self.pmwp(inputs, enable_pmwp_loss)
        
        if enable_pmwp_loss:
            return super().extract_feat(pmwp_out), pmwp_loss
        else:
            return super().extract_feat(pmwp_out)
    
    def loss(self, inputs: Tensor, data_samples):
        if self.enable_pmwp_loss:
            x, pmwp_loss = self.extract_feat(inputs, self.enable_pmwp_loss)
            losses:dict = pmwp_loss
        else:
            x = self.extract_feat(inputs)
            losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses


class AutoWindowStatusLoggerHook(Hook):
    def __init__(self, dpi:int=100):
        self.dpi = dpi

    def before_val_epoch(self, runner:Runner) -> None:
        model:ParalleledMultiWindowProcessing = runner.model.pmwp
        plt.figure(figsize=(4,3))
        WinE, VluP, CrsF = model.status() # Class `AutoWindowSetting`
        
        if (isinstance(runner._train_loop, dict)
                or runner._train_loop is None):
            current_iter = 0
        else:
            current_iter = runner.iter
        
        for k in list(WinE.keys()):
            if "RespMat" in k:
                signal, response = WinE.pop(k)
                buf = BytesIO()
                plt.clf()
                plt.plot(signal, response)
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=self.dpi)
                image = np.array(Image.open(buf).convert('RGB'))
                runner.visualizer.add_image(k, image, current_iter)
        
        runner.visualizer.add_scalars(
            WinE, 
            step=current_iter,
            file_path=f"{runner.timestamp}-AutoWindowStatus.json")
        
        if VluP is not None:
            for name, proj in VluP.items():
                plt.clf()
                buf = BytesIO()
                plt.plot(proj[0], proj[1])
                plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes)
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=self.dpi)
                image = np.array(Image.open(buf).convert('RGB'))
                runner.visualizer.add_image(name, image, current_iter)
        
        if CrsF is not None:
            plt.clf()
            buf = BytesIO()
            seaborn.heatmap(CrsF, cmap='rainbow')
            plt.savefig(buf, format='png')
            image = np.array(Image.open(buf).convert('RGB'))
            runner.visualizer.add_image("CrsF", image, current_iter)
