import os
import re
import argparse
import logging
import pdb
from bdb import BdbQuit
from pprint import pprint
from os import path as osp
from colorama import Fore, Style

from mmengine.logging import print_log
from mmengine.config import Config, DictAction
from mmengine.analysis import get_model_complexity_info
from mmengine.runner.checkpoint import find_latest_checkpoint

from mgamdata.mm.mmeng_PlugIn import DynamicRunnerSelection



DEFAULT_WORK_DIR = '/fileser51/zhangyiqin.sx/mmseg/work_dirs'
DEFAULT_CONFIG_DIR = '/home/zhangyq.sx/openmm/configs'
SUPPORTED_MODELS = ['MedNeXt']



def parase_args():
    parser = argparse.ArgumentParser(description="MMSEG")
    parser.add_argument("exp_name",             type=str,   nargs="+",          help="实验名")
    parser.add_argument("--VRamAlloc",          type=str,   default="pytorch",  help="设置内存分配器")
    parser.add_argument("--local-rank",         type=int,   default=0,          help="节点数量")
    parser.add_argument("--models",             type=str,   default=SUPPORTED_MODELS,   help="选择实验",    nargs="+",)
    parser.add_argument("--work_dir_root",      type=str,   default=DEFAULT_WORK_DIR,   help="存储实验结果的根目录") 
    parser.add_argument("--config_root",        type=str,   default=DEFAULT_CONFIG_DIR, help="存储配置文件的根目录")
    parser.add_argument('--cfg-options',        nargs='+',  action=DictAction)
    parser.add_argument("--test-draw-interval", type=int,   default=None,       help="测试时可视化样本的间距")
    parser.add_argument("--test",               default=False,  action='store_true', help="仅测试模式")
    parser.add_argument("--auto-retry",         type=int,   default=0,          help="单个实验出错自动重试次数")
    args = parser.parse_args()
    return args



def model_param_stat(cfg, runner):
    model = runner.model
    image_size = cfg.crop_size  # (256,256)
    analysis_results = get_model_complexity_info(model, input_shape=(cfg.in_chans, *image_size))
    print(analysis_results['out_table'], 
          file=open(osp.join(runner._log_dir, 'ModelParamStat_table.log'), 'w', encoding='utf-8'))
    print(analysis_results['out_arch'],
          file=open(osp.join(runner._log_dir, 'ModelParamStat_arch.log'), 'w', encoding='utf-8'))
    pprint(analysis_results,
           open(osp.join(runner._log_dir, 'ModelParamStat_all.log'), 'w', encoding='utf-8'))



def global_env_init(args):
    # 获取当前终端启动的路径
    current_path = os.getcwd()
    if os.path.basename(current_path)!="openmm":
        raise RuntimeError("当前终端路径异常，请确认在 openmm 目录下启动")
    # 异步内存分配器指定
    if args.VRamAlloc == "pytorch":
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"backend:native"
    elif args.VRamAlloc == "cuda":
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"backend:cudaMallocAsync"
    else:
        raise NotImplementedError



def VersionToFullExpName(name, config_root):
    if name[-1] == ".":
        raise AttributeError(f"目标实验名不得以“.”结尾：{name}")
    exp_list = os.listdir(config_root)
    for exp in exp_list:
        if exp == name:
            print(f"已找到实验：{name} <-> {exp}")
            return exp
        elif exp.startswith(name):
            pattern = r'\.[a-zA-Z]'    # 正则表达式找到第一次出现"."与字母连续出现的位置
            match = re.search(pattern, exp)
            if match is None:
                raise ValueError(f"在{config_root}目录下，无法匹配实验号：{exp}")
            if exp[:match.start()] == name:
                print(f"已根据实验号找到实验：{name} -> {exp}")
                return exp
            # if not "." in exp[len(name)+1:]:       # 序列号后方不得再有“.”
            #     if not exp[len(name)+1].isdigit(): # 序列号若接了数字，说明该目录实验是目标实验的子实验
                    # print(f"已根据实验号找到实验：{name} -> {exp}")
                    # return exp
    raise RuntimeError(f"未找到与“ {name} ”匹配的实验名")



def modify_cfg_to_set_visualization(cfg, work_dir, draw):
    default_hooks = cfg.default_hooks
    if draw:
        visualization_hook = default_hooks.get('visualization',None)
        # Turn on visualization
        if visualization_hook:
            visualization_hook['draw'] = True
        visualizer = cfg.visualizer
        if visualizer:
            visualizer['save_dir'] = work_dir
    return cfg



def modify_cfg_to_skip_train(config, draw_interval):
    # remove train and val cfgs
    config.train_dataloader = None
    config.train_cfg = None
    config.optim_wrapper = None
    config.param_scheduler = None
    config.val_dataloader = None
    config.val_cfg = None
    config.val_evaluator = None
    config.logger_interval = 10
    config.resume = False
    if draw_interval:
        config.default_hooks.visualization.interval = draw_interval
    else:
        config.default_hooks.visualization.draw = False
    return config



def modify_cfg_to_ensure_single_node(cfg):
    cfg.launcher = 'none'
    cfg.model_wrapper_cfg = None
    cfg.strategy = None
    cfg.Compile = None
    cfg.compile = None
    return cfg



class mmseg_experiment:
    def __init__(self, 
                 config, 
                 work_dir, 
                 test_draw_interval, 
                 cfg_options, 
                 test_mode):
        self.config = config
        self.work_dir = work_dir
        self.test_draw_interval = test_draw_interval
        self.cfg_options = cfg_options
        self.test_mode = test_mode
        self._prepare_basic_config()
        
        if self.IsTested(self.cfg):
            print_log(f"测试已经完成, 跳过: {self.work_dir}", 'current', logging.INFO)
        
        elif test_mode is True:
            self._direct_to_test()
            # model_param_stat(cfg, runner)
            print_log(f"测试完成: {work_dir}", 'current', logging.INFO)
        
        elif self.IsTrained(self.cfg):
            print_log(f"训练已经完成, 请在终端手动切换至单卡模式进行test: {self.work_dir}", 'current', logging.INFO)
            
        else:
            runner = DynamicRunnerSelection(self.cfg) # 建立Runner
            print_log(f"训练开始: {work_dir}", 'current', logging.INFO)
            runner.train()
            print_log(f"训练已经完成, 请在终端手动切换至单卡模式进行test: {self.work_dir}", 'current', logging.INFO)
            
            
    def _prepare_basic_config(self):
        cfg = Config.fromfile(self.config)  # load config
        cfg.work_dir = self.work_dir        # set work dir
        if self.cfg_options is not None:
            self.cfg = cfg.merge_from_dict(self.cfg_options)   # cfg override
        
        # 实验没有结束，初始化模型，调整mmseg配置参数
        print_log(f"启动中，初始化模型: {self.work_dir}", 'current', logging.INFO)
        self.cfg = modify_cfg_to_set_visualization(cfg, self.work_dir, self.test_draw_interval)
    
    
    def _direct_to_test(self):
        # 检查是否处于torchrun模式
        if os.getenv('LOCAL_RANK') is not None:
            print(f"Running with torchrun. Test mode requires single GPU mode.")
        
        cfg = modify_cfg_to_skip_train(self.cfg, self.test_draw_interval)
        cfg = modify_cfg_to_ensure_single_node(cfg)
        runner = DynamicRunnerSelection(cfg)
        ckpt_path = find_latest_checkpoint(self.work_dir)
        print_log(f"载入检查点: {self.work_dir}", 'current', logging.INFO)
        runner.load_checkpoint(ckpt_path)
        print_log(f"载入完成，执行测试: {self.work_dir}", 'current', logging.INFO)
        runner.test()
        # model_param_stat(cfg, runner) # 模型参数统计
        print_log(f"测试完成: {self.work_dir}", 'current', logging.INFO)
    
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



def auto_runner(args):
    for exp in args.exp_name:
        exp = VersionToFullExpName(exp, args.config_root)
        print(f"{exp} 实验启动")

        for model in args.models:
            # 确定配置文件路径和保存路径
            config_path = os.path.join(args.config_root, f"{exp}/{model}.py")
            work_dir_path = osp.join(args.work_dir_root, exp)
            # 设置终端标题
            if os.name == 'nt':
                os.system(f"{model} - {exp} ")
            else:
                print(f"\n--------- {model} - {exp} ---------\n")
            # 带有自动重试的执行
            remain_chance = args.auto_retry + 1
            while remain_chance:
                remain_chance -= 1
                try:
                    mmseg_experiment(config_path, 
                                     work_dir_path, 
                                     args.test_draw_interval, 
                                     args.cfg_options,
                                     args.test)
                
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except BdbQuit:
                    raise BdbQuit
                except Exception as e:
                    if remain_chance == 0:
                        print(Fore.RED + f"Exception，重试{args.auto_retry}失败，中止。错误原因:\n" + Style.RESET_ALL, e)
                        raise e
                    else:
                        print(Fore.YELLOW + f"Exception，剩余重试次数：{remain_chance}，错误原因:\n" + Style.RESET_ALL, e)
                else:
                    print(Fore.GREEN + f"实验完成: {work_dir_path}" + Style.RESET_ALL)
                    break



def main():
    args = parase_args()
    global_env_init(args)
    auto_runner(args)





if __name__ == '__main__':
    main()