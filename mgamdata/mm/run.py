import os
import re
import argparse
import pdb
from bdb import BdbQuit
from os import path as osp
from colorama import Fore, Style

from mmengine.config import DictAction

from mgamdata.mm import MM_WORK_DIR_ROOT, MM_TEST_DIR_ROOT, MM_CONFIG_ROOT

SUPPORTED_MODELS = os.environ['supported_models'].split(',')


class auto_runner:

    def __init__(self, exp_names, model_names, work_dir_root,
                 test_work_dir_root, config_root, cfg_options,
                 test_draw_interval, test, auto_retry):
        self.exp_names = exp_names
        self.model_names = model_names
        self.work_dir_root = work_dir_root
        self.test_work_dir_root = test_work_dir_root
        self.config_root = config_root
        self.cfg_options = cfg_options
        self.test_draw_interval = test_draw_interval
        self.test = test
        self.auto_retry = auto_retry

    @classmethod
    def start_from_args(cls):
        parser = argparse.ArgumentParser(description="暮光霭明的OpenMM实验运行器")
        parser.add_argument("exp_name", type=str, nargs="+", help="实验名或实验版本号")
        parser.add_argument("--VRamAlloc",
                            type=str,
                            default="pytorch",
                            help="设置内存分配器")
        parser.add_argument("--local-rank", type=int, default=0, help="节点数量")
        parser.add_argument("--models",
                            type=str,
                            default=SUPPORTED_MODELS,
                            help="选择实验",
                            nargs="+",
        )
        parser.add_argument("--work-dir-root",
                            type=str,
                            default=MM_WORK_DIR_ROOT,
                            help="存储实验结果的根目录")
        parser.add_argument("--test-work-dir-root",
                            type=str,
                            default=MM_TEST_DIR_ROOT,
                            help="测试时的工作目录")
        parser.add_argument("--config-root",
                            type=str,
                            default=MM_CONFIG_ROOT,
                            help="存储配置文件的根目录")
        parser.add_argument('--cfg-options', nargs='+', action=DictAction)
        parser.add_argument("--test-draw-interval",
                            type=int,
                            default=None,
                            help="测试时可视化样本的间距")
        parser.add_argument("--test",
                            default=False,
                            action='store_true',
                            help="仅测试模式")
        parser.add_argument("--auto-retry",
                            type=int,
                            default=0,
                            help="单个实验出错自动重试次数")
        args = parser.parse_args()

        return cls(exp_names=args.exp_name,
                   model_names=args.models,
                   work_dir_root=args.work_dir_root,
                   test_work_dir_root=args.test_work_dir_root,
                   config_root=args.config_root,
                   cfg_options=args.cfg_options,
                   test_draw_interval=args.test_draw_interval,
                   test=args.test,
                   auto_retry=args.auto_retry)

    def find_full_exp_name(self, exp_name):
        if exp_name[-1] == ".":
            raise AttributeError(f"目标实验名不得以“.”结尾：{exp_name}")

        exp_list = os.listdir(self.config_root)
        for exp in exp_list:

            if exp == exp_name:
                print(f"已找到实验：{exp_name} <-> {exp}")
                return exp

            elif exp.startswith(exp_name):
                pattern = r'\.[a-zA-Z]'  # 正则表达式找到第一次出现"."与字母连续出现的位置
                match = re.search(pattern, exp)

                if match is None:
                    raise ValueError(f"在{self.config_root}目录下，无法匹配实验号：{exp}")

                if exp[:match.start()] == exp_name:
                    print(f"已根据实验号找到实验：{exp_name} -> {exp}")
                    return exp

        else:
            raise RuntimeError(f"在 {MM_CONFIG_ROOT} 中未找到与“ {exp_name} ”匹配的实验名")

    def experiment_queue(self):
        print("实验队列启动, 正在import依赖...")
        from mgamdata.mm.experiment import experiment

        for exp in self.exp_names:
            exp = self.find_full_exp_name(exp)
            print(f"{exp} 实验启动")

            for model in self.model_names:
                # 确定配置文件路径和保存路径
                config_path = os.path.join(self.config_root,
                                           f"{exp}/{model}.py")
                work_dir_path = osp.join(self.work_dir_root, exp, model)
                test_work_dir_path = osp.join(self.test_work_dir_root, exp, model)

                # 设置终端标题
                if os.name == 'nt':
                    os.system(f"{model} - {exp} ")
                else:
                    print(f"\n--------- {model} - {exp} ---------\n")

                # 带有自动重试的执行
                remain_chance = self.auto_retry + 1
                while remain_chance:
                    remain_chance -= 1

                    try:
                        experiment(config_path, work_dir_path,
                                   test_work_dir_path, self.test_draw_interval,
                                   self.cfg_options, self.test)

                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except BdbQuit:
                        raise BdbQuit

                    except Exception as e:
                        if remain_chance == 0:
                            print(
                                Fore.RED +
                                f"Exception，重试{self.auto_retry}失败，中止。错误原因:\n" +
                                Style.RESET_ALL, e)
                            raise e
                        else:
                            print(
                                Fore.YELLOW +
                                f"Exception，剩余重试次数：{remain_chance}，错误原因:\n" +
                                Style.RESET_ALL, e)

                    else:
                        print(Fore.GREEN + f"实验完成: {work_dir_path}" +
                              Style.RESET_ALL)
                        break


def main():
    runner = auto_runner.start_from_args()
    runner.experiment_queue()


if __name__ == '__main__':
    main()
