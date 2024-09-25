import os
import os.path as osp
import pdb
import json
from colorama import Fore, Style
from multiprocessing import Pool
from tqdm import tqdm

import SimpleITK as sitk

from mgamdata.io.sitk_toolkit import (
    LoadDcmAsSitkImage, LoadDcmAsSitkImage, 
    sitk_resample_to_spacing_v2,
    sitk_resample_to_size)




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


if __name__ == '__main__':
    """
        本脚本用于执行dcm至mha的转换。
        
        DCM_AXIAL_SORT_MODE: 排序规则
            - engineering 工程提测所使用的排序法
            - JianYing 鉴影标注系统所使用的排序法
        src_dcm_root: dcm文件夹的根目录，每个病例的dcm文件夹应该在此目录下
        dest_mha_root: 转换后的mha文件保存的根目录
    """
    
    DCM_AXIAL_SORT_MODE = 'engineering'
    src_dcm_root  = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Test_7986/dcm'
    dest_mha_root = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Test_7986/mha_original_EngineerSort/image'
    spacing = None
    size = None
    
    confirm = input(f"{Fore.RED}WARNING, CONFIRM SORTING MODE: {Fore.YELLOW}{DCM_AXIAL_SORT_MODE}{Fore.RED}, then press enter.{Style.RESET_ALL}")
    if confirm != '':
        exit(0)
    
    with Pool(32) as p:
        failed = []
        results = []
        os.makedirs(dest_mha_root, exist_ok=True)
        patient_names = [folder for folder in os.listdir(src_dcm_root) 
                         if osp.isdir(osp.join(src_dcm_root, folder))]
        
        for patient_name in patient_names:
            src_folder = osp.join(src_dcm_root, patient_name)
            dst_path = osp.join(dest_mha_root, osp.basename(src_folder)+'.mha')
            new_task = p.apply_async(convert_one_case, 
                                     args=(DCM_AXIAL_SORT_MODE, src_folder, dst_path, spacing, size))
            results.append(new_task)
        
        for result in tqdm(results, desc='Converting', dynamic_ncols=True):
            result = result.get()
            if result is not None:
                failed.append(result)
    
    if len(failed) > 0:
        json.dump(failed, 
                  open(osp.join(dest_mha_root, 'failed.json'), 'w'), 
                  indent=4, 
                  ensure_ascii=False)
    
