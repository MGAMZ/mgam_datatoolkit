import os
import pdb
import torch
import datetime
import logging
from typing import Dict

import pandas as pd
from mmengine.runner import Runner, IterBasedTrainLoop
from mmengine.runner.runner import ConfigType
from mmengine.hooks import LoggerHook
from mmengine.registry import RUNNERS
from mmengine.logging import print_log
from .SA_Med2D import SA_Med2D_Dataset

"""
Runner在执行训练时是最为底层的实现，
目前还不支持Python风格的配置文件直接指定Runner，
因此在此采用传统的方法进行注册。
"""

@RUNNERS.register_module()
class mgam_Runner(Runner):
    def __init__(self, **kwargs):
        self.custom_env(kwargs.get('env_cfg', {}))
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
        torch.cuda.set_device(cfg.get('torch_cuda_id', 0))
        # Torch Compile
        cfg.get('torch_logging_level', logging.WARN)
        torch._logging.set_logs(all=self.str_to_log_level(cfg.get('torch_logging_level', 'WARN')))
        torch._logging.set_logs(dynamo=self.str_to_log_level(cfg.get('dynamo_logging_level', 'WARN')))
        torch._dynamo.config.cache_size_limit = cfg.get('dynamo_cache_size', 1) # type:ignore
        torch._dynamo.config.suppress_errors = cfg.get('dynamo_supress_errors', False) # type:ignore
        # cuBLAS matmul
        torch.backends.cuda.matmul.allow_tf32 = cfg.get('allow_tf32', False)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = \
            cfg.get('allow_fp16_reduced_precision_reduction', False)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = \
            cfg.get('allow_bf16_reduced_precision_reduction', True)
        # CUDNN
        torch.backends.cudnn.allow_tf32 = cfg.get('allow_tf32', False)
        torch.backends.cudnn.benchmark = cfg.get('benchmark', False)
        torch.backends.cudnn.deterministic = cfg.get('deterministic', False)

    @classmethod
    def from_cfg(cls, cfg):
        if  isinstance(cfg, ConfigType) and \
            issubclass(cfg.train_dataset.type, SA_Med2D_Dataset):
            
            union_atom_map, label_map, proxy, union_atom_map_path = \
            SA_Med2D_Dataset.set_proxy(cfg['modality'], cfg['dataset_source'], True)
            num_classes = len(label_map.keys())
            cfg = cls.auto_configure_num_classes_from_Databackend(cfg, num_classes)
        
        return super().from_cfg(cfg)
    
    @staticmethod
    def auto_configure_num_classes_from_Databackend(cfg:ConfigType, num_classes):
        for key, value in cfg.items():
            if key == 'num_classes' or key == 'out_channels':
                print_log(f"NumClasses Auto Override {cfg.get('type', 'Unknown')}: {cfg['num_classes']} -> {num_classes}",
                          'current')
                cfg[key] = num_classes
            elif isinstance(value, ConfigType):
                cfg[key] = mgam_Runner.auto_configure_num_classes_from_Databackend(value, num_classes)
        return cfg



class IterBasedTrainLoop_SupportProfiler(IterBasedTrainLoop):
    def __init__(self, profiler:str, *args, **kwargs):
        self.profiler = profiler
        self.profiler_step_count = 0
        super().__init__(*args, **kwargs)

        if profiler == 'PyTorchProfiler':
            from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
            self.prof = profile(
                activities=[ProfilerActivity.CPU, 
                            ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=50,
                    warmup=1,
                    active=2),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=True,
                with_modules=True,
                on_trace_ready=tensorboard_trace_handler('./work_dirs/profiler/'))
            self.prof.start()
    
    def run_iter(self, data_batch) -> None:
        if hasattr(self, 'prof'):
            super().run_iter(data_batch)
            self.prof.step()
            self.profiler_step_count += 1
            if self.profiler_step_count == 50+1+2:
                exit(-5)
        else:
            super().run_iter(data_batch)



class mgam_PerClassMetricLogger_OnTest(LoggerHook):
    def after_test_epoch(self,
                        runner,
                        metrics:Dict
                        ) -> None:
        PerClassResult_FromIoUMetric = metrics.pop('Perf/PerClass')
        data_df = pd.DataFrame(PerClassResult_FromIoUMetric)    # [Class, metrics...]
        # calculate average for each column except the first column
        data_df.loc['mean'] = data_df.iloc[:, 1:].mean(axis=0)
        data_df = data_df.round(decimals=2)
        csv_path_suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_save_path = os.path.join(runner.log_dir, f'PerClassResult_{csv_path_suffix}.csv')
        data_df.to_csv(csv_save_path, index=False)
        
        super().after_test_epoch(runner, metrics)



