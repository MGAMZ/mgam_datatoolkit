import os
import os.path as osp
import pdb
import datetime
import logging
import json
from functools import partial
from numbers import Number

import torch
import pandas as pd
import numpy as np
from torch import Tensor
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

from ..utils.DevelopUtils import measure_time, InjectVisualize


# support FSDP
def DynamicRunnerSelection(cfg: ConfigType) -> Runner:
    if cfg.use_FSDP:
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
            strategy = kwargs.get("cfg", {}).pop("strategy", None)

            if issubclass(self.__class__, FlexibleRunner):
                auto_strategy = partial(
                    size_based_auto_wrap_policy, min_num_params=int(1e7)
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
