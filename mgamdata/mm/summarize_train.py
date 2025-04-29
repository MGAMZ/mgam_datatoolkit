from functools import partial
import os
import glob
import json
import argparse
import matplotlib.pyplot as plt



def process_json_and_plot(json_file: str, output_file: str, exp_name: str, model_name: str, plt_params:dict|None=None):
    """
    处理单个JSON文件并绘制折线图
    
    Args:
        json_file: JSON文件路径
        output_file: 输出图像路径
        exp_name: 实验名称
        model_name: 模型名称
    """
    # 解析JSON文件
    records = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                print(f"警告: 无法解析 {json_file} 中的一行")
    
    if not records:
        print(f"警告: {json_file} 中没有有效记录，跳过")
        return
    
    # 检查每条记录，确保只有iter或epoch
    for i, record in enumerate(records):
        if 'iter' in record and 'epoch' in record:
            raise ValueError(f"错误: 在 {json_file} 的第 {i+1} 条记录中同时存在 iter 和 epoch")
    
    # 提取所有键，排除step、iter和epoch
    keys_to_plot = set()
    for record in records:
        for key in record.keys():
            if key not in ['step', 'iter', 'epoch']:
                keys_to_plot.add(key)
    
    # 转换为列表并排序
    keys_to_plot = sorted(list(keys_to_plot))
    
    # 提取step和其他指标的值
    steps = []
    metrics = {key: [] for key in keys_to_plot}
    
    for record in records:
        if 'step' in record:
            steps.append(record['step'])
            for key in keys_to_plot:
                if key in record:
                    metrics[key].append(record[key])
                else:
                    metrics[key].append(None)  # 使用None标记缺失值
    
    # 计算子图布局
    n_metrics = len(keys_to_plot)
    if n_metrics == 0:
        print(f"警告: {json_file} 中没有需要绘制的指标，跳过")
        return
    
    # 计算行列数
    n_cols = 1
    n_rows = n_metrics
    
    # 创建图形
    plt.figure(figsize=(8, 3 * n_rows))
    plt.suptitle(f"{exp_name}/{model_name}", fontsize=16)
    
    # 绘制每个指标的折线图
    for i, key in enumerate(keys_to_plot):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # 过滤掉None值
        valid_indices = [i for i, val in enumerate(metrics[key]) if val is not None]
        if not valid_indices:
            ax.text(0.5, 0.5, f"No data for {key}", ha='center', va='center')
            ax.set_title(key)
            continue
        
        valid_steps = [steps[i] for i in valid_indices]
        valid_values = [metrics[key][i] for i in valid_indices]
        
        # 检查是否所有值都是数值
        try:
            # 尝试转换为float
            valid_values = [float(v) for v in valid_values]
            ax.plot(valid_steps, valid_values, marker='.', linestyle='-', markersize=2)
        except (ValueError, TypeError):
            ax.text(0.5, 0.5, f"Non-numeric data for {key}", ha='center', va='center')
        
        ax.set_title(key)
        ax.set_xlabel("Step")
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为标题留出空间
    plt.savefig(output_file, dpi=400)
    plt.close()
    
    print(f"图像已保存到 {output_file}")


def parse_train_logs_to_figures(log_root: str, output_dir: str, exp_names=None, plt_params:dict|None=None):
    """
    从日志文件根目录遍历实验和模型文件夹，读取训练日志，绘制折线图并保存。
    
    Args:
        log_root: 日志文件根目录
        output_dir: 输出图像的目录
        exp_names: 如果指定，则只处理这些实验名
    """
    
    
    tasks = []
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    exp_list = sorted(os.listdir(log_root))
    
    # 如果指定了实验名，则进行过滤
    if exp_names:
        exp_list = [exp for exp in exp_list if exp in exp_names]
        if not exp_list:
            print(f"警告: 未找到指定的实验 {exp_names}")
            return tasks
    
    for exp_name in exp_list:
        exp_path = os.path.join(log_root, exp_name)
        if not os.path.isdir(exp_path):
            continue
        
        for model_name in sorted(os.listdir(exp_path)):
            model_path = os.path.join(exp_path, model_name)
            if not os.path.isdir(model_path):
                continue
            
            model_vis_data_path = os.path.join(model_path, "vis_data")
            
            # 在某些版本的实验中，json文件可能直接出现在模型目录下，而不是时间文件夹下
            json_files = [f for f in glob.glob(os.path.join(model_vis_data_path, "*.json")) 
                            if os.path.basename(f) != "scalars.json"]
            # 处理每个JSON文件
            print(f"正在处理 {exp_name}/{model_name}/，找到 {len(json_files)} 个训练日志文件")
            for json_file in json_files:
                json_filename = os.path.basename(json_file)
                tasks.append(partial(
                    process_json_and_plot, 
                    json_file=json_file, 
                    output_file=os.path.join(output_dir, f"{exp_name}_{model_name}_{json_filename.replace('.json','.png')}"),
                    exp_name=exp_name, 
                    model_name=model_name))
            
            # 进入时间文件夹
            for time_folder in sorted(os.listdir(model_path)):
                time_folder_path = os.path.join(model_path, time_folder)
                if not os.path.isdir(time_folder_path):
                    continue
                
                # 再进入vis_data文件夹
                vis_data_path = os.path.join(time_folder_path, "vis_data")
                if not os.path.isdir(vis_data_path):
                    continue
                
                # 查找所有JSON文件（排除scalars.json）
                json_files = [f for f in glob.glob(os.path.join(vis_data_path, "*.json")) 
                              if os.path.basename(f) != "scalars.json"]
                if not json_files:
                    continue
                
                print(f"正在处理 {exp_name}/{model_name}/{time_folder}，找到 {len(json_files)} 个训练日志文件")
                
                # 处理每个JSON文件
                for json_file in json_files:
                    json_filename = os.path.basename(json_file)
                    tasks.append(partial(
                        process_json_and_plot, 
                        json_file=json_file, 
                        output_file=os.path.join(output_dir, f"{exp_name}_{model_name}_{json_filename.replace('.json','.png')}"),
                        exp_name=exp_name, 
                        model_name=model_name,
                        plt_params=plt_params))

    return tasks


def plot_experiment_comparison(metric_name, model_name, exp_data, output_dir, plt_params:dict|None=None):
    """
    绘制多个实验中同一模型的指标对比图
    
    Args:
        metric_name: 指标名称
        model_name: 模型名称
        exp_data: 该模型下不同实验的数据 {实验名: [数据点]}
        output_dir: 输出目录
        plt_params: 绘图参数
    """
    plt.figure(figsize=(8, 4))
    
    # 为不同实验使用不同颜色
    colors = plt.cm.tab10.colors
    color_idx = 0
    
    # 遍历该模型下的所有实验数据
    for exp_name, data_points in sorted(exp_data.items()):
        if not data_points:
            continue
        
        # 排序数据点
        data_points.sort(key=lambda x: x[0])
        steps, values = zip(*data_points)
        
        label = f"{exp_name}"  # 只显示实验名，因为模型名已在标题中
        plt.plot(steps, values, marker='.', linestyle='-', 
                markersize=2, label=label, color=colors[color_idx % len(colors)])
        color_idx += 1
    
    plt.title(f"{model_name} - {metric_name}", fontsize=14)
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 处理图例显示
    if plt_params is None or plt_params.get("plt_no_legend", False) is False:
        # 将图例放在整个图的底部
        plt.legend(loc='best')
    elif plt_params.get("plt_no_legend", True):
        plt.legend().set_visible(False)
    
    # 调整布局，为底部的图例留出空间
    plt.tight_layout()
    
    # 文件名中替换可能的非法字符
    safe_metric_name = metric_name.replace('/', '_').replace('\\', '_')
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    output_file = os.path.join(output_dir, f"compare_{safe_model_name}_{safe_metric_name}.png")
    
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存到 {output_file}")


def compare_experiments_and_plot(log_root: str, output_dir: str, exp_names: list, plt_params:dict|None=None):
    """
    对比多个实验的相同指标，将它们按模型分组绘制在不同图上
    
    Args:
        log_root: 日志文件根目录
        output_dir: 输出图像的目录
        exp_names: 要对比的实验名列表
        plt_params: 绘图参数
    """
    
    def collect_metrics_from_json(json_file, all_metrics, exp_name, model_name):
        """从JSON文件中收集指标数据"""
        records = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析 {json_file} 中的一行")
        
        if not records:
            return
        
        # 提取所有指标
        for record in records:
            if 'step' not in record:
                continue
                
            step = record['step']
            for key, value in record.items():
                if key in ['step', 'iter', 'epoch']:
                    continue
                    
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue
                    
                # 添加到全局指标字典 - 按指标名、模型名、实验名分组
                if key not in all_metrics:
                    all_metrics[key] = {}
                if model_name not in all_metrics[key]:
                    all_metrics[key][model_name] = {}
                if exp_name not in all_metrics[key][model_name]:
                    all_metrics[key][model_name][exp_name] = []
                    
                all_metrics[key][model_name][exp_name].append((step, value))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 收集所有实验的数据 - 新结构: {指标名: {模型名: {实验名: [数据点]}}}
    all_metrics = {}
    # 获取所有实验中出现的模型名称集合
    all_models = set()
    
    # 收集数据
    for exp_name in exp_names:
        exp_path = os.path.join(log_root, exp_name)
        if not os.path.isdir(exp_path):
            print(f"警告: 找不到实验目录 {exp_name}")
            continue
        
        for model_name in sorted(os.listdir(exp_path)):
            model_path = os.path.join(exp_path, model_name)
            if not os.path.isdir(model_path):
                continue
                
            all_models.add(model_name)
            
            # 处理模型目录下的JSON文件
            model_vis_data_path = os.path.join(model_path, "vis_data")
            json_files = [f for f in glob.glob(os.path.join(model_vis_data_path, "*.json")) 
                          if os.path.basename(f) != "scalars.json"]
            
            for json_file in json_files:
                collect_metrics_from_json(json_file, all_metrics, exp_name, model_name)
            
            # 处理时间文件夹下的JSON文件
            for time_folder in sorted(os.listdir(model_path)):
                time_folder_path = os.path.join(model_path, time_folder)
                if not os.path.isdir(time_folder_path):
                    continue
                
                vis_data_path = os.path.join(time_folder_path, "vis_data")
                if not os.path.isdir(vis_data_path):
                    continue
                
                json_files = [f for f in glob.glob(os.path.join(vis_data_path, "*.json")) 
                              if os.path.basename(f) != "scalars.json"]
                
                for json_file in json_files:
                    collect_metrics_from_json(json_file, all_metrics, exp_name, model_name)
    
    # 为每个指标的每个模型创建对比图
    tasks = []
    for metric_name, models_data in all_metrics.items():
        for model_name, exp_data in models_data.items():
            # 检查该模型下至少有两个实验的数据
            if len(exp_data) < 2:
                continue
                
            tasks.append(partial(
                plot_experiment_comparison,
                metric_name=metric_name,
                model_name=model_name,
                exp_data=exp_data,
                output_dir=output_dir,
                plt_params=plt_params
            ))
    
    print(f"找到 {len(all_models)} 个模型和 {len(all_metrics)} 个指标，生成 {len(tasks)} 个对比图")
    return tasks


def run_tasks(tasks, mp:bool=False):
    from tqdm import tqdm
    if mp:
        from multiprocessing import Pool
        with Pool() as pool:
            mp_proc = [pool.apply_async(task) for task in tasks]
            for mp_task in tqdm(mp_proc, 
                                desc="Processing tasks", 
                                dynamic_ncols=True):
                mp_task.get()
    else:
        for task in tqdm(tasks, desc="Processing tasks"):
            task()


def main():
    parser = argparse.ArgumentParser(description="解析训练日志并绘制折线图")
    parser.add_argument("log_root", type=str, help="日志文件根目录")
    parser.add_argument("output_dir", type=str, help="输出图像的目录")
    parser.add_argument("--mp", action="store_true", help="是否使用多进程处理任务")
    parser.add_argument("--exp", nargs='+', help="指定要对比的实验名（可指定多个）")
    parser.add_argument("--plt-no-legend", action="store_true", default=False, help="不显示图例")
    args = parser.parse_args()
    
    plt_params = {
        "plt_no_legend": args.plt_no_legend
    }
    
    
    if args.exp:
        print(f"对比模式: 将比较以下实验: {', '.join(args.exp)}")
        tasks = compare_experiments_and_plot(args.log_root, args.output_dir, args.exp, plt_params)
    else:
        tasks = parse_train_logs_to_figures(args.log_root, args.output_dir, plt_params)
        
    if not tasks:
        print("没有找到任何训练日志文件，退出。")
        return
        
    run_tasks(tasks, mp=args.mp)



if __name__ == "__main__":
    main()