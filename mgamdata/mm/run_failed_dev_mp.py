import os
import sys
import re
import argparse
import logging
import pdb
import subprocess
from bdb import BdbQuit
from pprint import pprint
from os import path as osp
from colorama import Fore, Style
from typing import List, Dict, Tuple

from mmengine.logging import print_log
from mmengine.config import Config, DictAction
from mmengine.analysis import get_model_complexity_info
from mmengine.runner.checkpoint import find_latest_checkpoint

from mgamdata.mm.mmeng_PlugIn import DynamicRunnerSelection


DEFAULT_WORK_DIR = '/fileser51/zhangyiqin.sx/mmseg/work_dirs'
DEFAULT_TEST_WORK_DIR = '/fileser51/zhangyiqin.sx/mmseg/test_work_dirs'
DEFAULT_CONFIG_DIR = '/root/mgam/openmm/configs'
SUPPORTED_MODELS = ['MedNeXt']



class experiment:
    def __init__(self,
                 config:str,
                 work_dir:str,
                 test_work_dir:str,
                 test_draw_interval:int,
                 cfg_options:Dict|None,
                 test_mode:bool):
        self.config = config
        self.work_dir = work_dir
        self.test_work_dir = test_work_dir
        self.test_draw_interval = test_draw_interval
        self.cfg_options = cfg_options
        self.test_mode = test_mode
        self._prepare_basic_config()
        
        if self.IsTested(self.cfg):
            print_log(f"{Fore.BLUE}测试已经完成, 跳过: {self.work_dir}{Style.RESET_ALL}", 'current', logging.INFO)
        
        elif test_mode is True:
            self._direct_to_test()
            # model_param_stat(cfg, runner)
            print_log(f"{Fore.GREEN}测试完成: {work_dir}{Style.RESET_ALL}", 'current', logging.INFO)
        
        elif self.IsTrained(self.cfg):
            print_log(f"{Fore.GREEN}训练已经完成: {self.work_dir}{Style.RESET_ALL}", 'current', logging.INFO)
            
        else:
            runner = DynamicRunnerSelection(self.cfg) # 建立Runner
            print_log(f"{Fore.BLUE}训练开始: {work_dir}{Style.RESET_ALL}", 'current', logging.INFO)
            runner.train()
            print_log(f"{Fore.GREEN}训练已经完成, 请在终端手动切换至单卡模式进行test: {self.work_dir}{Style.RESET_ALL}", 'current', logging.INFO)

        exit(0)


    def _prepare_basic_config(self):
        cfg = Config.fromfile(self.config)  # load config
        cfg.work_dir = self.work_dir        # set work dir
        if self.cfg_options is not None:
            cfg = cfg.merge_from_dict(self.cfg_options)   # cfg override
        
        # 实验没有结束，初始化模型，调整mmseg配置参数
        print_log(f"启动中，初始化模型: {self.work_dir}", 'current', logging.INFO)
        self.cfg = cfg
        self.modify_cfg_to_set_visualization()


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
        ckpt_path = find_latest_checkpoint(self.work_dir)
        print_log(f"载入检查点: {self.work_dir}", 'current', logging.INFO)
        runner.load_checkpoint(ckpt_path)
        print_log(f"载入完成，执行测试: {self.work_dir}", 'current', logging.INFO)
        
        # 执行测试
        runner.test()
        
        # model_param_stat(cfg, runner) # 模型参数统计


    def modify_cfg_to_set_visualization(self):
        default_hooks = self.cfg.default_hooks
        if self.test_draw_interval:
            visualization_hook = default_hooks.get('visualization',None)
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
        last_ckpt = open(os.path.join(work_dir_path, "last_checkpoint"), 'r').read()
        last_ckpt = re.findall(r"iter_(\d+)", last_ckpt)[0].strip(r'iter_')
        if int(last_ckpt) >= target_iters:
            return True
        else:
            return False

    @staticmethod
    def IsTested(cfg:str) -> bool:
        test_file_path = os.path.join(
            cfg.work_dir, 
            f"test_result_epoch{cfg.get('epoch', 0)}_iter{cfg.get('iters', 0)}.json")
        if os.path.exists(test_file_path): 
            return True
        else:
            return False



class auto_runner:
    def __init__(self,
                 exp_names: List[str],
                 model_names: List[str],
                 work_dir_root: str,
                 test_work_dir_root: str,
                 config_root: str,
                 cfg_options: Dict|None,
                 test_draw_interval: int,
                 test: bool,
                 auto_retry: int,
                 multiprocess: bool
        ):
        self.exp_names = exp_names
        self.model_names = model_names
        self.work_dir_root = work_dir_root
        self.test_work_dir_root = test_work_dir_root
        self.config_root = config_root
        self.cfg_options = cfg_options
        self.test_draw_interval = test_draw_interval
        self.test = test
        self.auto_retry = auto_retry
        self.multiprocess = multiprocess
        
        self.experiment_queue() # 启动实验队列


    def find_full_exp_name(self, exp_name):
        if exp_name[-1] == ".":
            raise AttributeError(f"目标实验名不得以“.”结尾：{exp_name}")
        
        exp_list = os.listdir(self.config_root)
        for exp in exp_list:
            
            if exp == exp_name:
                print(f"已找到实验：{exp_name} <-> {exp}")
                return exp
            
            elif exp.startswith(exp_name):
                pattern = r'\.[a-zA-Z]'    # 正则表达式找到第一次出现"."与字母连续出现的位置
                match = re.search(pattern, exp)
                
                if match is None:
                    raise ValueError(f"在{self.config_root}目录下，无法匹配实验号：{exp}")
                
                if exp[:match.start()] == exp_name:
                    print(f"已根据实验号找到实验：{exp_name} -> {exp}")
                    return exp
                
                # if not "." in exp[len(name)+1:]:       # 序列号后方不得再有“.”
                #     if not exp[len(name)+1].isdigit(): # 序列号若接了数字，说明该目录实验是目标实验的子实验
                        # print(f"已根据实验号找到实验：{name} -> {exp}")
                        # return exp
        
        raise RuntimeError(f"未找到与“ {exp_name} ”匹配的实验名")


    def experiment_queue(self):
        for exp in self.exp_names:
            exp = self.find_full_exp_name(exp)
            print(Fore.BLUE, f"{exp} 实验启动", Style.RESET_ALL)

            for model in self.model_names:
                # 确定配置文件路径和保存路径
                config_path = os.path.join(self.config_root, f"{exp}/{model}.py")
                work_dir_path = osp.join(self.work_dir_root, exp)
                test_dir_path = osp.join(self.test_work_dir_root, exp)
                
                # 设置终端标题
                if os.name == 'nt':
                    os.system(Fore.BLUE, f"\n****** {model} - {exp} ******\n", Style.RESET_ALL)
                else:
                    print(Fore.BLUE, f"\n****** {model} - {exp} ******\n", Style.RESET_ALL)
                
                # 带有自动重试的执行
                remain_chance = self.auto_retry + 1
                while remain_chance:
                    remain_chance -= 1
                    
                    try:
                        start_cmd_args = [
                            'python',
                            __file__,
                            'exp',
                            config_path,
                            work_dir_path,
                            test_dir_path,
                            f'--test-draw-interval={self.test_draw_interval}',
                        ]
                        if self.cfg_options is not None:
                            if len(self.cfg_options) > 0:
                                start_cmd_args += ['--cfg-options']
                                for key, value in self.cfg_options.items():
                                    start_cmd_args += [f"{key}={value}"]
                        if self.test is True:
                            start_cmd_args += ['--test']
                        
                        if self.multiprocess:
                            self.multi_node_call(start_cmd_args)
                        else:
                            self.single_node_call(start_cmd_args)
                        
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    
                    except BdbQuit:
                        raise BdbQuit
                    
                    except Exception as e:
                        if remain_chance == 0:
                            print(Fore.RED + f"Exception! 重试{self.auto_retry}次后依旧失败。")
                            print("错误原因:\n", e, Style.RESET_ALL)
                            if isinstance(e, subprocess.CalledProcessError):
                                print(Fore.RED,
                                      "\n\n检测到来自于子进程的错误\n子进程错误原因 Original Traceback:\n",
                                      e.stderr,
                                      Style.RESET_ALL)
                                
                                print(Fore.RED, "子进程启动指令：\n")
                                print(" ".join(e.cmd), Style.RESET_ALL, '\n')
                            raise e
                        
                        else:
                            print(Fore.YELLOW + f"Exception! 剩余重试次数：{remain_chance}")
                            print("错误原因:\n", e, Style.RESET_ALL)
                            if isinstance(e, subprocess.CalledProcessError):
                                print(Fore.YELLOW,
                                      "\n\n检测到来自于子进程的错误\n子进程错误原因 Original Traceback:\n",
                                      e.stderr,
                                      Style.RESET_ALL)
                    
                    else:
                        print(Fore.GREEN + f"实验完成: {work_dir_path}" + Style.RESET_ALL)
                        break


    def single_node_call(self, start_cmd_args):
        subprocess.run(
            args=' '.join(start_cmd_args),
            shell=True,
            check=True,
        )


    def multi_node_call(self, start_cmd_args):
        from torch.cuda import device_count
        start_cmd_args[0] = f'python -m torch.distributed.launch --nproc_per_node={device_count()}'
        # start_cmd_args = ['torchrun', '--nproc-per-node', str(device_count())] + start_cmd_args
        subprocess.run(
            args=' '.join(start_cmd_args),
            shell=True,
            check=True,
        )



def parse_args():
    parser = argparse.ArgumentParser(description="MGAM - OPENMM - EXECUTOR")
    subparser = parser.add_subparsers(dest='command')
    parser_run = subparser.add_parser('run', help='对外运行接口')
    parser_exp = subparser.add_parser('exp', help='对内调用接口')
    
    parser_run.add_argument("exp_names",            type=str,   nargs="+",                      help="实验名")
    parser_run.add_argument("--VRamAlloc",          type=str,   default="pytorch",              help="设置内存分配器")
    parser_run.add_argument("--local-rank",         type=int,   default=0,                      help="节点数量")
    parser_run.add_argument("--models",             type=str,   default=SUPPORTED_MODELS,       help="选择实验",    nargs="+",)
    parser_run.add_argument("--work-dir-root",      type=str,   default=DEFAULT_WORK_DIR,       help="存储实验结果的根目录")
    parser_run.add_argument("--test-work-dir-root", type=str,   default=DEFAULT_TEST_WORK_DIR,  help="测试时的工作目录")
    parser_run.add_argument("--config-root",        type=str,   default=DEFAULT_CONFIG_DIR,     help="存储配置文件的根目录")
    parser_run.add_argument("--test-draw-interval", type=int,   default=0,                      help="测试时可视化样本的间距")
    parser_run.add_argument("--auto-retry",         type=int,   default=0,                      help="单个实验出错自动重试次数")
    parser_run.add_argument("--cfg-options",        nargs='+',      action=DictAction)
    parser_run.add_argument("--mp",                 default=False,  action='store_true', help="多卡模式")
    parser_run.add_argument("--test",               default=False,  action='store_true', help="仅测试模式")
    
    parser_exp.add_argument("config",               type=str,       help="配置文件路径")
    parser_exp.add_argument("work_dir",             type=str,       help="工作目录")
    parser_exp.add_argument("test_work_dir",        type=str,       help="测试工作目录")
    parser_exp.add_argument("--test-draw-interval", type=int,       default=0,           help="测试时可视化样本的间距")
    parser_exp.add_argument("--cfg-options",        nargs='+',      action=DictAction)
    parser_exp.add_argument("--test",               default=False,  action='store_true', help="仅测试模式")
    
    return parser.parse_args()



def select_mode_and_call(args):
    if args.command == 'run':
        auto_runner(args.exp_names,
                    args.models,
                    args.work_dir_root,
                    args.test_work_dir_root,
                    args.config_root,
                    args.cfg_options,
                    args.test_draw_interval,
                    args.test,
                    args.auto_retry,
                    args.mp)
    
    elif args.command == 'exp':
        experiment(args.config, 
                args.work_dir, 
                args.test_work_dir, 
                args.test_draw_interval, 
                args.cfg_options, 
                args.test)
    
    return 0



def main():
    args = parse_args()

    try:
        select_mode_and_call(args)
        if args.test is not True:
            args.test = True
            select_mode_and_call(args)
    
    except Exception as e:
        raise e




if __name__ == '__main__':
    main()

