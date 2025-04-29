import os
import os.path as osp
import re
import pdb
import glob
import logging
from colorama import Fore, Style

import torch

from mmengine.logging import print_log
from mmengine.config import Config
from mmengine.runner.checkpoint import find_latest_checkpoint

from mgamdata.mm.mmeng_PlugIn import DynamicRunnerSelection


class experiment:

    def __init__(self, 
                 config, 
                 work_dir, 
                 test_work_dir, 
                 cfg_options, 
                 test_mode, 
                 detect_anomaly, 
                 test_use_last_ckpt):
        self.config = config
        self.work_dir = work_dir
        self.test_work_dir = test_work_dir
        self.cfg_options = cfg_options
        self.test_mode = test_mode
        self.detect_anomaly = detect_anomaly
        self.test_use_last_ckpt = test_use_last_ckpt
        
        with torch.autograd.set_detect_anomaly(detect_anomaly):
            self._prepare_basic_config()
            self._main_process()

    def _main_process(self):
        if self.IsTested(self.cfg):
            print_log(
                f"{Fore.BLUE}测试已经完成, 跳过: {self.work_dir}{Style.RESET_ALL}",
                'current', logging.INFO)

        elif self.test_mode is True:
            print_log(f"{Fore.BLUE}测试开始: {self.work_dir}{Style.RESET_ALL}",
                      'current', logging.INFO)
            self._direct_to_test()
            # model_param_stat(cfg, runner)
            print_log(f"{Fore.GREEN}测试完成: {self.work_dir}{Style.RESET_ALL}",
                      'current', logging.INFO)

        elif self.IsTrained(self.cfg):
            print_log(
                f"{Fore.BLUE}训练已经完成, 请在终端手动切换至单卡模式进行test: {self.work_dir}{Style.RESET_ALL}",
                'current', logging.INFO)

        else:
            runner = DynamicRunnerSelection(self.cfg)  # 建立Runner
            print_log(f"{Fore.BLUE}训练开始: {self.work_dir}{Style.RESET_ALL}",
                      'current', logging.INFO)
            runner.train()
            print_log(
                f"{Fore.GREEN}训练已经完成, 请在终端手动切换至单卡模式进行test: {self.work_dir}{Style.RESET_ALL}",
                'current', logging.INFO)

    def _prepare_basic_config(self):
        cfg = Config.fromfile(self.config)  # load config
        cfg.work_dir = self.work_dir  # set work dir
        if self.cfg_options is not None:
            cfg = cfg.merge_from_dict(self.cfg_options)  # cfg override

        # 实验没有结束，初始化模型，调整mmseg配置参数
        print_log(f"启动中，初始化模型: {self.work_dir}", 'current', logging.INFO)
        self.cfg = cfg

    def _direct_to_test(self):
        # 检查是否处于torchrun模式
        if os.getenv('LOCAL_RANK') is not None:
            print(f"Running with torchrun. Test mode requires single GPU mode.")

        # 配置文件调整到test模式
        self.modify_cfg_to_skip_train()
        self.modify_cfg_to_ensure_single_node()
        self.modify_cfg_to_set_test_work_dir()

        # 模型初始化
        runner = DynamicRunnerSelection(self.cfg)
        if self.test_use_last_ckpt:
            ckpt_path = find_latest_checkpoint(self.work_dir)
        else:
            ckpt_path = glob.glob(osp.join(self.work_dir, 'best*.pth'))
            assert len(ckpt_path) == 1, f"尝试在 {ckpt_path} 找到最佳模型，但不能确定最佳。"
            ckpt_path = ckpt_path[0]
        print_log(f"载入检查点: {self.work_dir}", 'current', logging.INFO)
        runner.load_checkpoint(ckpt_path)
        print_log(f"载入完成，执行测试: {self.work_dir}", 'current', logging.INFO)

        # 执行测试
        runner.test()

        # model_param_stat(cfg, runner) # 模型参数统计
        print_log(f"测试完成: {self.work_dir}", 'current', logging.INFO)

    def modify_cfg_to_skip_train(self):
        # remove train and val cfgs
        self.cfg.train_dataloader = None
        self.cfg.train_cfg = None
        self.cfg.optim_wrapper = None
        self.cfg.param_scheduler = None
        self.cfg.val_dataloader = None
        self.cfg.val_cfg = None
        self.cfg.val_evaluator = None
        self.cfg.resume = False

    def modify_cfg_to_ensure_single_node(self):
        self.cfg.launcher = 'none'
        self.cfg.model_wrapper_cfg = None
        self.cfg.strategy = None
        self.cfg.Compile = None
        self.cfg.compile = None

    def modify_cfg_to_set_test_work_dir(self):
        self.cfg.work_dir = self.test_work_dir
        self.cfg.visualizer.save_dir = self.test_work_dir

    @staticmethod
    def IsTrained(cfg) -> bool:
        if "iters" in cfg.keys():
            target_iters = cfg.iters
            work_dir_path = cfg.work_dir
            if not os.path.exists(os.path.join(work_dir_path, "last_checkpoint")):
                return False
            last_ckpt = open(os.path.join(work_dir_path, "last_checkpoint"),
                            'r').read()
            last_ckpt = re.findall(r"iter_(\d+)", last_ckpt)[0].strip(r'iter_')
        else:
            target_iters = cfg.epochs
            work_dir_path = cfg.work_dir
            if not os.path.exists(os.path.join(work_dir_path, "last_checkpoint")):
                return False
            last_ckpt = open(os.path.join(work_dir_path, "last_checkpoint"),
                            'r').read()
            last_ckpt = re.findall(r"epoch_(\d+)", last_ckpt)[0].strip(r'epoch_')
        if int(last_ckpt) >= target_iters:
            return True
        else:
            return False

    @staticmethod
    def IsTested(cfg: str) -> bool:
        test_file_path = os.path.join(
            cfg.work_dir,
            f"test_result_epoch{cfg.get('epoch', 0)}_iter{cfg.get('iters', 0)}.json"
        )
        if os.path.exists(test_file_path):
            return True
        else:
            return False
