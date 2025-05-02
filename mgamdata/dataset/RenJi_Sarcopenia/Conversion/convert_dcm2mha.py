import os
import os.path as osp
import argparse
import pdb
import json
from colorama import Fore, Style
from multiprocessing import Pool
from tqdm import tqdm

import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import (
    LoadDcmAsSitkImage, sitk_resample_to_spacing_v2, sitk_resample_to_size)




# NOTE 鉴影的读取顺序和工程的读取顺序不同
def convert_one_case(sort_mode, src_folder, dst_path, spacing, size):
    try:
        assert ((spacing is not None) and (size is not None)) is False, \
            "Spacing and Size can not be set at the same time."
        if osp.exists(dst_path):
            return
        
        image, _, _, _ = LoadDcmAsSitkImage(sort_mode, src_folder, spacing=None)
        
        if spacing is not None:
            image = sitk_resample_to_spacing_v2(image, spacing, 'image')
        if size is not None:
            image = sitk_resample_to_size(image, size, 'image')
        
        sitk.WriteImage(image, dst_path, useCompression=True)
    
    except Exception as e:
        return {
            'failed_path': src_folder,
            'error': str(e)
        }



def parse_args():
    parser = argparse.ArgumentParser(description='Convert DCM to MHA')
    parser.add_argument('sort_mode', type=str,
                        choices=['engineering', 'JianYing'], 
                        help='Z轴排序法则，鉴影系统和工程实现的排序规则是不同的。')
    parser.add_argument('src_dcm_root', type=str,
                        help='包含所有DCM序列文件夹的父文件夹')
    parser.add_argument('dest_mha_root', type=str,
                        help='保存转换后的MHA文件的文件夹')
    parser.add_argument('--spacing', type=str, default=None, 
                        help='可以在转换时就对mha进行一次重采样，依据spacing')
    parser.add_argument('--size', type=str, default=None, 
                        help='可以在转换时就对mha进行一次重采样，依据size')
    parser.add_argument('--mp', action='store_true', 
                        help='使用多进程加速转换')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    
    confirm = input(f"{Fore.YELLOW}WARNING, CONFIRM SORTING MODE: {Fore.YELLOW}{args.sort_mode}{Fore.RED}, then press enter.{Style.RESET_ALL}")
    if confirm != '':
        exit(0)
    
    failed = []
    results = []
    os.makedirs(args.dest_mha_root, exist_ok=True)
    patient_names = [folder 
                    for folder in os.listdir(args.src_dcm_root) 
                    if osp.isdir(osp.join(args.src_dcm_root, folder))]
    
    if args.mp is True:
        with Pool(32) as p:
            for patient_name in patient_names:
                src_folder = osp.join(args.src_dcm_root, patient_name)
                dst_path = osp.join(args.dest_mha_root, osp.basename(src_folder)+'.mha')
                new_task = p.apply_async(convert_one_case, 
                                        args=(args.sort_mode, src_folder, dst_path, args.spacing, args.size))
                results.append(new_task)
            
            for result in tqdm(results, desc='Converting', dynamic_ncols=True):
                result = result.get()
                if result is not None:
                    failed.append(result)

    else:
        for patient_name in tqdm(patient_names, desc='Converting', dynamic_ncols=True):
            src_folder = osp.join(args.src_dcm_root, patient_name)
            dst_path = osp.join(args.dest_mha_root, osp.basename(src_folder)+'.mha')
            result = convert_one_case(args.sort_mode, src_folder, dst_path, args.spacing, args.size)
            if result is not None:
                failed.append(result)
    
    if len(failed) > 0:
        json.dump(failed, 
                  open(osp.join(args.dest_mha_root, 'failed.json'), 'w'), 
                  indent=4, 
                  ensure_ascii=False)

