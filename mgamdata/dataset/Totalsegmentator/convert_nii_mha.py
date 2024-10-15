import os
import argparse
import shutil
import json
import pdb
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import SimpleITK as sitk
from monai.data.image_reader import NibabelReader



def convert_nii_file(args):
    nii_file_path, save_root, data_root = args
    try:
        # 构建保存路径，保持原有的文件夹结构
        relative_path = os.path.relpath(os.path.dirname(nii_file_path), data_root)
        save_dir = os.path.join(save_root, relative_path)
        os.makedirs(save_dir, exist_ok=True)
        mha_file_path = os.path.join(save_dir, os.path.basename(nii_file_path).replace('.nii.gz', '.mha'))
        if os.path.exists(mha_file_path):
            return None
        
        # 读取 nii.gz 文件
        try:
            image = sitk.ReadImage(nii_file_path)
        except:
            monai_nii_reader = NibabelReader()
            nii_image = monai_nii_reader.read(nii_file_path)
            image, meta_dict = monai_nii_reader.get_data(nii_image)
            pdb.set_trace()
            
        # 保存为 mha 文件
        sitk.WriteImage(image, mha_file_path, useCompression=True)
        return None  # 成功时返回 None
    
    except Exception as e:
        raise e
        return {'file': nii_file_path, 'error': str(e)}  # 失败时返回错误信息



def convert_folder(data_root, save_root, use_multiprocessing):
    # 收集所有的 nii.gz 文件路径
    nii_files = []
    failed_files = []
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
            else:
                # direct copy
                relative_path = os.path.relpath(root, data_root)
                save_dir = os.path.join(save_root, relative_path)
                os.makedirs(save_dir, exist_ok=True)
                shutil.copy(os.path.join(root, file), os.path.join(save_dir, file))
    
    if use_multiprocessing:
        # 使用多进程处理
        with Pool(cpu_count()) as pool:
            for result in tqdm(
                iterable=pool.imap_unordered(
                    convert_nii_file,
                    [(nii_file, save_root, data_root) for nii_file in nii_files]),
                total=len(nii_files),
                desc="Converting files",
                dynamic_ncols=True):
                
                if result:
                    failed_files.append(result)
    
    else:
        # 单进程处理
        for nii_file in tqdm(
            iterable=nii_files,
            desc="Converting files",
            dynamic_ncols=True):
            
            result = convert_nii_file((nii_file, save_root, data_root))
            if result:
                failed_files.append(result)
    
    # 保存失败记录到 failed.json
    if failed_files:
        with open(os.path.join(save_root, 'failed.json'), 'w') as f:
            json.dump(failed_files, f, indent=4)



def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI (.nii.gz) files to MetaImage (.mha) format.")
    parser.add_argument('data_root', type=str, help="Root directory containing .nii.gz files.")
    parser.add_argument('save_root', type=str, help="Directory to save converted .mha files.")
    parser.add_argument('--mp', action='store_true', help="Enable multiprocessing.")
    
    args = parser.parse_args()
    
    convert_folder(args.data_root, args.save_root, args.mp)




if __name__ == "__main__":
    main()