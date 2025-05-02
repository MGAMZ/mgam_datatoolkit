import os
import argparse
import multiprocessing as mp
import functools
from pathlib import Path
from tqdm import tqdm

import nibabel as nib


def fix_nifti_file(input_path, output_path):
    """修复 NIfTI 文件的 qform 和 sform"""
    try:
        img = nib.load(input_path)
        qform = img.get_qform()
        img.set_qform(qform)
        sform = img.get_sform()
        img.set_sform(sform)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, output_path)
        return True
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {str(e)}")
        return False


def process_file(file_path, input_root, output_root):
    """处理单个文件并保持目录结构"""
    # 确保所有参数都是 Path 对象
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not isinstance(input_root, Path):
        input_root = Path(input_root)
    if not isinstance(output_root, Path):
        output_root = Path(output_root)
    
    # 计算相对路径，以便在输出目录中保持相同的结构
    try:
        rel_path = file_path.relative_to(input_root)
        output_path = output_root / rel_path
        return fix_nifti_file(file_path, output_path)
    except ValueError as e:
        print(f"计算相对路径时出错: {str(e)}")
        return False


def find_nifti_files(root_dir):
    """递归查找所有 .nii.gz 文件"""
    root_path = Path(root_dir)
    return list(root_path.glob('**/*.nii.gz'))


def main():
    parser = argparse.ArgumentParser(description='修复 NIfTI 文件的 qform 和 sform')
    parser.add_argument('input_root', help='输入目录根路径')
    parser.add_argument('output_root', help='输出目录根路径')
    parser.add_argument('--mp', type=int, default=0, help='使用的进程数量，0 表示不使用多进程')
    
    args = parser.parse_args()
    
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    
    # 验证输入路径
    if not input_root.exists():
        print(f"错误: 输入目录 '{input_root}' 不存在")
        exit(1)
    
    # 确保输出根目录存在
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 .nii.gz 文件
    nifti_files = find_nifti_files(input_root)
    file_count = len(nifti_files)
    print(f"找到 {file_count} 个 NIfTI 文件")
    
    if file_count == 0:
        print("未找到任何 NIfTI 文件，退出程序")
        exit(0)
    
    if args.mp > 0:
        # 使用多进程处理
        num_processes = min(args.mp, mp.cpu_count())
        print(f"使用 {num_processes} 个进程进行处理")
        
        with mp.Pool(processes=num_processes) as pool:
            process_func = functools.partial(process_file, input_root=input_root, output_root=output_root)
            results = list(tqdm(pool.imap_unordered(process_func, nifti_files), total=file_count,
                                desc="处理文件",
                                dynamic_ncols=True))
        
        success_count = sum(1 for result in results if result)
    else:
        # 使用单进程处理
        print("使用单进程处理")
        success_count = 0
        for file_path in tqdm(nifti_files,
                              desc="处理文件",
                              dynamic_ncols=True):
            if process_file(file_path, input_root, output_root):
                success_count += 1
    
    print(f"处理完成，成功修复 {success_count}/{file_count} 个文件")



if __name__ == "__main__":
    main()