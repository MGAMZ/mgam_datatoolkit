import os
import os.path as osp
import re
import logging
from colorama import Fore, Style

from mmengine.logging import print_log
from mmengine.config import Config
from mmengine.analysis import get_model_complexity_info
from mmengine.runner.checkpoint import find_latest_checkpoint

from mgamdata.mm.mmeng_PlugIn import DynamicRunnerSelection


class experiment:

    def __init__(self, config, work_dir, test_work_dir, test_draw_interval,
                 cfg_options, test_mode):
        self.config = config
        self.work_dir = work_dir
        self.test_work_dir = test_work_dir
        self.test_draw_interval = test_draw_interval
        self.cfg_options = cfg_options
        self.test_mode = test_mode
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
        self.modify_cfg_to_set_visualization()

    def _direct_to_test(self):
        # 检查是否处于torchrun模式
        if os.getenv('LOCAL_RANK') is not None:
            print(
                f"Running with torchrun. Test mode requires single GPU mode.")

        # 配置文件调整到test模式
        self.modify_cfg_to_skip_train()
        self.modify_cfg_to_ensure_single_node()
        self.modify_cfg_to_set_test_work_dir()

        # 模型初始化
        runner = DynamicRunnerSelection(self.cfg)
        ckpt_path = find_latest_checkpoint(self.work_dir)
        print_log(f"载入检查点: {self.work_dir}", 'current', logging.INFO)
        runner.load_checkpoint(ckpt_path)
        print_log(f"载入完成，执行测试: {self.work_dir}", 'current', logging.INFO)

        # 执行测试
        runner.test()

        # model_param_stat(cfg, runner) # 模型参数统计
        print_log(f"测试完成: {self.work_dir}", 'current', logging.INFO)

    def modify_cfg_to_set_visualization(self):
        default_hooks = self.cfg.default_hooks
        if self.test_draw_interval:
            visualization_hook = default_hooks.get('visualization', None)
            # Turn on visualization
            if visualization_hook:
                visualization_hook['draw'] = True
            if self.get('visualizer', None):
                self.cfg.visualizer['save_dir'] = self.work_dir

    def modify_cfg_to_skip_train(self):
        # remove train and val cfgs
        self.cfg.train_dataloader = None
        self.cfg.train_cfg = None
        self.cfg.optim_wrapper = None
        self.cfg.param_scheduler = None
        self.cfg.val_dataloader = None
        self.cfg.val_cfg = None
        self.cfg.val_evaluator = None
        self.cfg.logger_interval = 10
        self.cfg.resume = False
        if self.test_draw_interval:
            self.cfg.default_hooks.visualization.interval = self.test_draw_interval
        else:
            self.cfg.default_hooks.visualization.draw = False

    def modify_cfg_to_ensure_single_node(self):
        self.cfg.launcher = 'none'
        self.cfg.model_wrapper_cfg = None
        self.cfg.strategy = None
        self.cfg.Compile = None
        self.cfg.compile = None

    def modify_cfg_to_set_test_work_dir(self):
        self.cfg.work_dir = self.test_work_dir

    @staticmethod
    def IsTrained(cfg) -> bool:
        target_iters = cfg.iters
        work_dir_path = cfg.work_dir
        if not os.path.exists(os.path.join(work_dir_path, "last_checkpoint")):
            return False
        last_ckpt = open(os.path.join(work_dir_path, "last_checkpoint"),
                         'r').read()
        last_ckpt = re.findall(r"iter_(\d+)", last_ckpt)[0].strip(r'iter_')
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
