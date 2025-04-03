import os
import pdb
import argparse
import glob
import json
import pandas as pd


def parse_test_results_to_xlsx(log_root: str, output_xlsx: str = "test_results.xlsx"):
    """
    从日志文件根目录中定位形如 test_result*.json 的文件（每个模型文件夹只能有一个），
    解析后将结果转换为 XLSX 多级列形式并保存。
    第一级文件夹名作为实验名，第二级文件夹名作为模型名。
    """
    
    # 用于暂存所有行记录，每条记录对应一个 (experiment, model, metrics...) 的结果
    records = []
    
    # 遍历第一层（实验名）和第二层（模型名）
    for exp_name in sorted(os.listdir(log_root)):
        exp_path = os.path.join(log_root, exp_name)
        if not os.path.isdir(exp_path):
            continue
        
        for model_name in sorted(os.listdir(exp_path)):
            model_path = os.path.join(exp_path, model_name)
            if not os.path.isdir(model_path):
                continue
            
            # 查找 test_result*.json 文件（只能有一个，否则报错）
            json_files = glob.glob(os.path.join(model_path, "test_result*.json"))
            if len(json_files) == 0:
                continue  # 未找到则跳过
            if len(json_files) > 1:
                raise ValueError(f"在路径 {model_path} 中发现多个 test_result*.json 文件，请检查。")
            
            json_file = json_files[0]
            
            # 读取 JSON
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 准备一个行记录
            row_dict = {}
            row_dict["Experiment"] = exp_name
            row_dict["Model"] = model_name
            
            # 处理非 PerClass 的部分
            # 例如 Perf/aAcc, Perf/mIoU 等，这些 key 可能会根据文件不同而变化
            for key, val in data.items():
                # 跳过 PerClass
                if key == "Perf/PerClass":
                    continue
                # 记录在行中，确保是基础类型
                row_dict[key] = val
            
            # 处理 PerClass
            # PerClass 包含多个类，每个类又包含多个动态 metric
            per_class_data = data.get("Perf/PerClass", {})
            class_names = per_class_data.get("Class", [])
            
            # 从 PerClass 中找到所有 metric 名（除 "Class"）
            # 比如 IoU, Acc, Dice, Fscore, Precision, Recall 等
            metrics = [m for m in per_class_data.keys() if m != "Class"]
            
            # 遍历每个类，写入相应 metric
            for idx, cls_name in enumerate(class_names):
                for metric in metrics:
                    metric_vals = per_class_data[metric]
                    if idx < len(metric_vals):
                        # 将 "类名|Metric" 合并成一个唯一键
                        row_dict[f"{cls_name}|{metric}"] = metric_vals[idx]
            
            records.append(row_dict)
    
    # 如果没有任何数据，则不必输出
    if not records:
        print("未找到任何 test_result*.json 文件或数据为空。")
        return
    
    # 用 DataFrame 承载结果
    df = pd.DataFrame(records)
    
    # 一般可以将 "类名|Metric" 解析为多级列
    # 先保存列名顺序，保证 ['Experiment', 'Model', ...] 在前面
    col_order = [c for c in df.columns if c not in ["Experiment", "Model"]]
    df = df[["Experiment", "Model"] + col_order]
    
    # 将电子表格中的 "类名|Metric" 转换为多级表头
    # 形如 ("Cancer", "IoU"), ("Cancer", "Acc") 等
    new_cols = []
    for c in df.columns:
        if c == "Experiment":
            new_cols.append(("Experiment", ""))  # 第一行显示 “Experiment”，第二行留空即可合并
        elif c == "Model":
            new_cols.append(("Model", ""))       # 第一行显示 “Model”，第二行留空即可合并
        elif "|" in c:  # PerClass
            cls_name, metric_name = c.split("|", 1)
            new_cols.append((cls_name, metric_name))
        else:  # 全局 (非 PerClass)
            new_cols.append(("Global", c))

    df.columns = pd.MultiIndex.from_tuples(new_cols)
    df.to_excel(output_xlsx, merge_cells=True)
    print(f"结果已导出到 {output_xlsx}")


def main():
    parser = argparse.ArgumentParser(description="Parse test results to XLSX.")
    parser.add_argument("log_root", type=str, help="Root directory of log files.")
    parser.add_argument("output_xlsx", type=str, help="Output XLSX file.")
    args = parser.parse_args()
    
    parse_test_results_to_xlsx(args.log_root, args.output_xlsx)


if __name__ == "__main__":
    main()
