import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import SimpleITK as sitk
import pandas as pd

from mgamdata.dataset.Totalsegmentator import TSD_CLASS_INDEX_MAP as CLASS_INDEX_MAP


def analyze_single_file(args):
    file_path, class_names = args
    img = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(img)
    unique_ids = set(int(i) for i in set(arr.flatten()))
    row = [1 if CLASS_INDEX_MAP[name] in unique_ids else 0 for name in class_names]
    return row


def analyze_mha_classes(directory, use_mp=False):
    """
    分析目录下所有mha文件中存在的类，返回二维数组。
    行：每个mha文件，列：所有类名（顺序同CLASS_INDEX_MAP）。
    """
    mha_files = [f for f in os.listdir(directory) if f.endswith('.mha')]
    mha_paths = [os.path.join(directory, f) for f in mha_files]
    class_names = list(CLASS_INDEX_MAP.keys())
    result = []

    if use_mp:
        with Pool(cpu_count()) as pool:
            for row in tqdm(pool.imap(analyze_single_file, [(path, class_names) for path in mha_paths]), total=len(mha_paths), desc="多进程分析中"):
                result.append(row)
    else:
        for file_path in tqdm(mha_paths, desc="单进程分析中"):
            result.append(analyze_single_file((file_path, class_names)))

    return class_names, mha_files, result


def main():
    parser = argparse.ArgumentParser(description="分析mha标签文件中的类别并保存为xlsx")
    parser.add_argument('label_dir', type=str, help='mha标签文件所在根目录')
    parser.add_argument('save_xlsx', type=str, help='结果保存的xlsx路径')
    parser.add_argument('--mp', action='store_true', help='是否使用多进程处理')
    args = parser.parse_args()

    class_names, mha_files, matrix = analyze_mha_classes(args.label_dir, use_mp=args.mp)
    df = pd.DataFrame(matrix, columns=class_names, index=mha_files)
    df.to_excel(args.save_xlsx)
    print(f"分析完成，结果已保存到 {args.save_xlsx}")


if __name__ == '__main__':
    main()