import os
import os.path as osp
import pdb
import datetime
import logging
import json
import copy
from functools import partial
from numbers import Number
from typing_extensions import Sequence

import torch
import pandas as pd
import numpy as np
from torch import Tensor, nn
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from mmengine.dataset.sampler import DefaultSampler
from mmengine.runner import (
    Runner,
    IterBasedTrainLoop,
    FlexibleRunner,
    find_latest_checkpoint,
)
from mmengine.runner.runner import ConfigType
from mmengine.hooks import LoggerHook
from mmengine.hooks import RuntimeInfoHook as _RuntimeInfoHook
from mmengine.logging import print_log, MMLogger
from mmengine.optim.optimizer import AmpOptimWrapper
from mmengine.model.wrappers import (
    MMDistributedDataParallel,
    MMFullyShardedDataParallel,
)
from mmengine.model.averaged_model import BaseAveragedModel
from mmengine.dataset.utils import default_collate
from mmengine._strategy.fsdp import FSDPStrategy

from ..utils.DevelopUtils import measure_time, InjectVisualize


# support FSDP
def DynamicRunnerSelection(cfg: ConfigType) -> Runner:
    if cfg.dist is True and cfg.MP_mode != "ddp":
        RunnerChoice = FlexibleRunner
    else:
        RunnerChoice = Runner

    class mgam_Runner(RunnerChoice):  # type: ignore
        """MGAM Customized MMEngine Runner"""

        def __init__(self, **kwargs):
            self.resume_optimizer = kwargs.get("cfg", {}).pop("resume_optimizer", True)
            self.resume_param_scheduler = kwargs.get("cfg", {}).pop(
                "resume_param_scheduler", True
            )
            self.custom_env(kwargs.get("env_cfg", {}))

            if cfg.MP_mode == "fsdp":
                strategy = kwargs.get("cfg", {}).pop("strategy", None)
                auto_strategy = partial(
                    size_based_auto_wrap_policy, 
                    min_num_params=int(1e7),
                    recurse=True,
                )
                strategy.update(
                    dict(model_wrapper=dict(auto_wrap_policy=auto_strategy))
                )
                kwargs["strategy"] = strategy
                kwargs["cfg"]["strategy"] = strategy

            super().__init__(**kwargs)

        @staticmethod
        def str_to_log_level(string):
            idx = getattr(logging, string.upper(), None)
            if idx is None:
                raise ValueError(f"Unsupported log level: {string}")
            else:
                return idx

        def custom_env(self, cfg):
            # Avoid device clash with OpenCV
            torch.cuda.set_device(cfg.pop("torch_cuda_id", 0))
            # Torch Compile
            cfg.get("torch_logging_level", logging.WARN)
            torch._logging.set_logs(
                all=self.str_to_log_level(cfg.pop("torch_logging_level", "WARN"))
            )
            torch._logging.set_logs(
                dynamo=self.str_to_log_level(cfg.pop("dynamo_logging_level", "WARN"))
            )
            torch._dynamo.config.cache_size_limit = cfg.pop(
                "dynamo_cache_size", 1
            )  # type:ignore
            torch._dynamo.config.suppress_errors = cfg.pop(
                "dynamo_supress_errors", False
            )  # type:ignore
            # cuBLAS matmul
            torch.backends.cuda.matmul.allow_tf32 = cfg.get("allow_tf32", False)
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = cfg.pop(
                "allow_fp16_reduced_precision_reduction", False
            )
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = cfg.pop(
                "allow_bf16_reduced_precision_reduction", True
            )
            # CUDNN
            torch.backends.cudnn.allow_tf32 = cfg.pop("allow_tf32", False)
            torch.backends.cudnn.benchmark = cfg.pop("benchmark", False)
            torch.backends.cudnn.deterministic = cfg.pop("deterministic", False)

        @staticmethod
        def auto_configure_num_classes_from_Databackend(cfg: ConfigType, num_classes):
            for key, value in cfg.items():
                if key == "num_classes" or key == "out_channels":
                    print_log(
                        f"NumClasses Auto Override {cfg.get('type', 'Unknown')}: {cfg['num_classes']} -> {num_classes}",
                        "current",
                    )
                    cfg[key] = num_classes
                elif isinstance(value, ConfigType):
                    cfg[key] = mgam_Runner.auto_configure_num_classes_from_Databackend(
                        value, num_classes
                    )
            return cfg

        def load_or_resume(self) -> None:
            """Load or resume checkpoint."""
            if self._has_loaded:
                return None

            # decide to load from checkpoint or resume from checkpoint
            resume_from = None
            if self._resume and self._load_from is None:
                # auto resume from the latest checkpoint
                resume_from = find_latest_checkpoint(self.work_dir)
                self.logger.info(
                    f"Auto resumed from the latest checkpoint {resume_from}."
                )
            elif self._resume and self._load_from is not None:
                # resume from the specified checkpoint
                resume_from = self._load_from

            if resume_from is not None:
                self.resume(
                    filename=resume_from,
                    resume_optimizer=self.resume_optimizer,
                    resume_param_scheduler=self.resume_param_scheduler,
                )
                self._has_loaded = True
            elif self._load_from is not None:
                self.load_checkpoint(self._load_from)
                self._has_loaded = True
    
    return mgam_Runner.from_cfg(cfg)


# for debug
class IterBasedTrainLoop_SupportProfiler(IterBasedTrainLoop):

    def __init__(self, profiler: str, *args, **kwargs):
        self.profiler = profiler
        self.profiler_step_count = 0
        super().__init__(*args, **kwargs)

        if profiler == "PyTorchProfiler":
            from torch.profiler import (
                profile,
                ProfilerActivity,
                tensorboard_trace_handler,
            )

            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=50, warmup=1, active=2),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=True,
                with_modules=True,
                on_trace_ready=tensorboard_trace_handler("./work_dirs/profiler/"),
            )
            self.prof.start()

    def run_iter(self, data_batch) -> None:
        if hasattr(self, "prof"):
            super().run_iter(data_batch)
            self.prof.step()
            self.profiler_step_count += 1
            if self.profiler_step_count == 50 + 1 + 2:
                exit(-5)
        else:
            super().run_iter(data_batch)


# support for better class-wise performance logging
class mgam_PerClassMetricLogger_OnTest(LoggerHook):

    def after_test_epoch(self, runner, metrics: dict) -> None:
        PerClassResult_FromIoUMetric = metrics.pop("Perf/PerClass")
        data_df = pd.DataFrame(PerClassResult_FromIoUMetric)  # [Class, metrics...]
        # calculate average for each column except the first column
        data_df.loc["mean"] = data_df.iloc[:, 1:].mean(axis=0)
        data_df = data_df.round(decimals=2)
        csv_path_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_save_path = osp.join(
            runner.log_dir, f"PerClassResult_{csv_path_suffix}.csv"
        )
        data_df.to_csv(csv_save_path, index=False)

        super().after_test_epoch(runner, metrics)


class LoggerJSON(LoggerHook):

    @staticmethod
    def _itemize_metric_(metrics):
        if isinstance(metrics, (Tensor, np.ndarray)):
            return metrics.tolist()
        elif isinstance(metrics, Number):
            return metrics
        elif isinstance(metrics, dict):
            for k in metrics.keys():
                metrics[k] = LoggerJSON._itemize_metric_(metrics[k])
        elif isinstance(metrics, list):
            for i in range(len(metrics)):
                metrics[i] = LoggerJSON._itemize_metric_(metrics[i])
        elif isinstance(metrics, tuple):
            metrics = list(metrics)
            for i in range(len(metrics)):
                metrics[i] = LoggerJSON._itemize_metric_(metrics[i])

        return metrics

    def after_test_epoch(self, runner, metrics: dict) -> None:
        self._itemize_metric_(metrics)
        json_save_path = osp.join(
            runner.work_dir,
            f"test_result_epoch{runner.cfg.get('epochs', 0)}_iter{runner.cfg.get('iters', 0)}.json",
        )

        with open(json_save_path, "w") as f:
            json.dump(metrics, f, indent=4)

        super().after_test_epoch(runner, metrics)


# better AMP support
class AmpPatchAccumulateOptimWarpper(AmpOptimWrapper):

    def update_params(  # type: ignore
        self,
        loss: torch.Tensor,
        step_kwargs: dict | None = None,
        zero_kwargs: dict | None = None,
        should_update: bool = True,
    ) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if should_update:
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)


# customized DDP training for our task.
class RemasteredDDP(MMDistributedDataParallel):
    """
    The official MMEngine's Distributed Model Wrapper makes none sense to me.
    So I override the following three methods, avoiding the warpper to influence
    the model's data flow design.
    """

    def train_step(self, *args, **kwargs):
        return self.module.train_step(*args, **kwargs)

    def val_step(self, *args, **kwargs):
        return self.module.val_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.module.test_step(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.module, name)

# customized FSDP training for our task.
class RemasteredFSDP(MMFullyShardedDataParallel):
    """
    The official MMEngine's Distributed Model Wrapper makes none sense to me.
    So I override the following three methods, avoiding the warpper to influence
    the model's data flow design.
    """

    def train_step(self, *args, **kwargs):
        return self.module.train_step(*args, **kwargs)

    def val_step(self, *args, **kwargs):
        return self.module.val_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.module.test_step(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.module, name)


from mmengine.registry import FUNCTIONS, MODEL_WRAPPERS
from mmengine.model import BaseDataPreprocessor, is_model_wrapper
from mmengine.device import get_device
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
class RemasteredFSDP_Strategy(FSDPStrategy):
    def _wrap_model(self, model: nn.Module) -> None:
        """Wrap the model to :obj:``MMFullyShardedDataParallel`` or other
        custom fully sharded data parallel module wrappers.

        Args:
            model (nn.Module): Model to be wrapped.

        Returns:
            FullyShardedDataParallel: ``MMFullyShardedDataParallel``
            or subclass of ``FullyShardedDataParallel``.
        """
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
                apply_activation_checkpointing  # noqa: E501
        except ImportError:
            apply_activation_checkpointing = None

        for module in model.modules():
            if isinstance(module, BaseDataPreprocessor):
                module.to(get_device())

        if is_model_wrapper(model):
            return

        if self.model_wrapper is None:
            self.model_wrapper = dict(type='MMFullyShardedDataParallel')

        default_args = dict(
            module=model,
            device_id=int(os.environ['LOCAL_RANK']),
            type='MMFullyShardedDataParallel')
        model = MODEL_WRAPPERS.build(
            self.model_wrapper, default_args=default_args)
        
        set_state_dict(
            model, 
            self.optim_state_dict(), 
            model_state_dict=self.model_state_dict(),
            optim_state_dict=self.optim_state_dict(),
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
        # model.set_state_dict_type(model, self.state_dict_type,
        #                           self.state_dict_config,
        #                           self.optim_state_dict_config)

        if self.activation_checkpointing is not None:
            if apply_activation_checkpointing is None:
                raise RuntimeError(
                    'activation_checkpointing maybe deprecated by current '
                    'PyTorch version, maybe you could switch to PyTorch 2.0 '
                    'or 2.1 to use `activation_checkpointing`.')
            cfg = copy.deepcopy(self.activation_checkpointing)
            with FUNCTIONS.switch_scope_and_registry(None):
                check_fn = cfg.pop('check_fn')
                if isinstance(check_fn, str):
                    check_fn = FUNCTIONS.get(check_fn)
                elif isinstance(check_fn, dict):
                    fn_type = check_fn.pop('type')
                    if isinstance(fn_type, str):
                        fn_type = FUNCTIONS.get(fn_type)
                    check_fn = partial(fn_type, **cfg)

                if not callable(check_fn):
                    raise TypeError('`check_fn` must be a callable function')
                apply_activation_checkpointing(model, check_fn=check_fn, **cfg)
        return model


class RatioSampler(DefaultSampler):
    """随机激活一定比例的样本"""

    def __init__(self, use_sample_ratio: float, **kwargs):
        super().__init__(**kwargs)
        self.use_sample_ratio = use_sample_ratio
        print_log(
            f"RatioSampler used, original num of batches {super().__len__()} -> used {len(self)}",
            MMLogger.get_current_instance(),
        )

    def __iter__(self):
        indices = np.array(list(super().__iter__()))
        num_samples = int(len(indices) * self.use_sample_ratio)
        sampled_indices = np.random.choice(indices, num_samples, replace=False)
        return iter(sampled_indices.tolist())

    def __len__(self):
        return int(super().__len__() * self.use_sample_ratio)


class RuntimeInfoHook(_RuntimeInfoHook):
    def after_train_iter(
        self, runner: Runner, batch_idx: int, data_batch: dict, outputs: dict
    ) -> None:
        if outputs is not None:
            for key, value in outputs.items():
                if 'loss' in key:
                    runner.message_hub.update_scalar(f"train/{key}", value)


def multi_sample_collate(data_batch: Sequence[dict]):
    """
    Compatible with `SampleAugment` Transform Class.
    This collate is to facilitate multi-sub-sample generation
    from the same sample.
    
    NOTE
    The reason to do SampleWiseInTimeAugment is the time comsumption
    for IO of an entire sample is too expensive, so it's better
    to augment the sample in time, thus accquiring multiple trainable sub-samples.
    """
    
    flattened = []
    for item in data_batch:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    data_batch = flattened

    return default_collate(data_batch)


class MomentumAvgModel(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 100,
                 interval: int = 1,
                 device: torch.device|None = None,
                 update_buffers: bool = False) -> None:
        super().__init__()
        
        # 检查分布式环境
        self.is_distributed = hasattr(model, 'module')
        self.is_deepspeed = hasattr(model, 'module') and hasattr(model.module, 'deepspeed')
        
        # DeepSpeed环境下获取完整模型
        if self.is_deepspeed:
            with model.module.summon_full_params():
                self.module = copy.deepcopy(model.module).requires_grad_(False)
        else:
            target_model = model.module if self.is_distributed else model
            self.module = copy.deepcopy(target_model).requires_grad_(False)
            
        self.interval = interval
        if device is not None:
            self.module = self.module.to(device)
            
        self.register_buffer('steps',
                           torch.tensor(0, dtype=torch.long, device=device))
                           
        self.update_buffers = update_buffers
        if update_buffers:
            state_dict = self.module.state_dict()
            self.avg_parameters = {
                k: v for k, v in state_dict.items() 
                if v.numel() > 0
            }
        else:
            params = dict(self.module.named_parameters())
            self.avg_parameters = {
                k: v for k, v in params.items() 
                if v.numel() > 0
            }
            
        # 动量参数检查
        assert 0.0 < momentum < 1.0, f'momentum must be in range (0.0, 1.0) but got {momentum}'
        if momentum > 0.5:
            print_log(
                'The value of momentum in EMA is usually a small number,'
                'which is different from the conventional notion of '
                f'momentum but got {momentum}. Please make sure the '
                f'value is correct.',
                logger='current', 
                level=logging.WARNING)
        self.momentum = momentum
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)

    def _get_current_param(self):
        if self.update_buffers:
            return self.module.state_dict()
        else:
            return dict(self.module.named_parameters())
    
    def update_parameters(self, model: nn.Module) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = (
            model.state_dict()
            if self.update_buffers else dict(model.named_parameters()))
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:  # type: ignore
            for k, p_avg in self.avg_parameters.items():
                # NOTE handle deepspeed model shred issue, p_avg may be empty here.
                if p_avg.dtype.is_floating_point and p_avg.shape==src_parameters[k].data.shape:
                    device = p_avg.device
                    self.avg_func(p_avg.data,
                                  src_parameters[k].data.to(device),
                                  self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data.to(b_avg.device))
        self.steps += 1  # type: ignore

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using the linear
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = max(self.momentum,
                       self.gamma / (self.gamma + self.steps.item()))
        averaged_param.lerp_(source_param, momentum)
