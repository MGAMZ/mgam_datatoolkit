from multiprocessing import Pool
import pydicom
import SimpleITK
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import cv2
import json
from tqdm import tqdm


from aitrox.utils.sitk_toolkit import sitk_resample_to_image


def min_max_scale(img):
    max_val = np.max(img)
    min_val = np.min(img)
    res_img = (img - min_val) / (max_val - min_val)
    return res_img * 255


def get_dcm_file(dcm):
    pixel_array = dcm.pixel_array
    pixel_array = pixel_array.astype(np.float32)
    if hasattr(dcm,'RescaleIntercept') and hasattr(dcm,'RescaleSlope'):
        ri = dcm.RescaleIntercept
        rs = dcm.RescaleSlope
        pixel_array = pixel_array * np.float32(rs) + np.float32(ri)
    
    wc = dcm.WindowCenter
    ww = dcm.WindowWidth
    res = np.clip(pixel_array, wc-ww//2, wc+ww//2)
    res = min_max_scale(res)
    res = res.astype('uint8')
    return res


def get_mask_2d(src_folder, dst_folder, cat, l3_csv):
    save_dir = f'{dst_folder}/mask_2d_new'    #img、mask     
    mask_dir = f'{src_folder}/mask_'
    instance_dir = f'{src_folder}/pic_'
    
    os.makedirs(save_dir, exist_ok=True)
    data = pd.read_csv(l3_csv)
    data_len = len(data)

    for i in np.arange(data_len):
        ss_data_i = data.iloc[i]
        slide_item = ss_data_i['序列编号']

        if not os.path.exists(os.path.join(mask_dir, slide_item)):
            continue

        os.makedirs(os.path.join(save_dir, slide_item), exist_ok=True)

        l3_start = int(ss_data_i['L3节段起始层数'])
        l3_end = int(ss_data_i['L3节段终止层数'])

        dcm_list = glob.glob(os.path.join(instance_dir, slide_item, '*.dcm'))
        dcm_list.sort()
        
        ydj_itk = SimpleITK.ReadImage(os.path.join(mask_dir, slide_item, slide_item + '_0.mha') )
        mask_ydj = SimpleITK.GetArrayFromImage(ydj_itk)
        
        otherggj_itk = SimpleITK.ReadImage(os.path.join(mask_dir, slide_item, slide_item + '_1.mha') )
        mask_otherggj = SimpleITK.GetArrayFromImage(otherggj_itk)
        
        pxzf_itk = SimpleITK.ReadImage(os.path.join(mask_dir, slide_item, slide_item + '_2.mha') )
        mask_pxzf = SimpleITK.GetArrayFromImage(pxzf_itk)
        
        nzzf_itk = SimpleITK.ReadImage(os.path.join(mask_dir, slide_item, slide_item + '_3.mha') )
        mask_nzzf = SimpleITK.GetArrayFromImage(nzzf_itk)
        
        dcm_len = len(dcm_list)
        
        for idx in range(l3_start-1, l3_end):
            img_ydj_mask = mask_ydj[dcm_len - 1 - idx]
            img_otherggj_mask = mask_otherggj[dcm_len - 1 - idx]
            img_pxzf_mask = mask_pxzf[dcm_len - 1 - idx]
            img_nzzf_mask = mask_nzzf[dcm_len -1 -idx]
            
            merge_mask = np.zeros_like(img_ydj_mask, dtype=np.uint8)

            if cat == 0:
                merge_mask[img_ydj_mask==1]=1
            elif cat == 1:
                merge_mask[img_otherggj_mask==1]=1
            elif cat == 2:
                merge_mask[img_pxzf_mask==1]=1
            elif cat == 3:
                merge_mask[img_nzzf_mask==1]=1
            elif cat == 4:
                merge_mask[img_ydj_mask==1]=1
                merge_mask[img_otherggj_mask==1]=2
                merge_mask[img_pxzf_mask==1]=3
                merge_mask[img_nzzf_mask==1]=4
        
            cv2.imwrite(os.path.join(save_dir, slide_item , f'{idx:06d}' + '.png'), merge_mask)


def process_mask_one_case(src_folder, dst_folder, image_folder, cat, patient_name):
    try:
        ydj_itk = SimpleITK.ReadImage(os.path.join(src_folder, patient_name, '0.mha') )
        mask_ydj = SimpleITK.GetArrayFromImage(ydj_itk)
        
        otherggj_itk = SimpleITK.ReadImage(os.path.join(src_folder, patient_name, '1.mha') )
        mask_otherggj = SimpleITK.GetArrayFromImage(otherggj_itk)
        
        pxzf_itk = SimpleITK.ReadImage(os.path.join(src_folder, patient_name, '2.mha') )
        
        mask_pxzf = SimpleITK.GetArrayFromImage(pxzf_itk)
        
        nzzf_itk = SimpleITK.ReadImage(os.path.join(src_folder, patient_name, '3.mha') )
        mask_nzzf = SimpleITK.GetArrayFromImage(nzzf_itk)
        
        mask_len = len(mask_ydj)
        if mask_len != len(mask_otherggj) or mask_len != len(mask_pxzf) or mask_len != len(mask_nzzf):
            raise RuntimeError('mask length not equal')
            
        data_mask_list = []
        for idx in np.arange(mask_len):
            img_ydj_mask = mask_ydj[idx]
            img_otherggj_mask = mask_otherggj[idx]
            img_pxzf_mask = mask_pxzf[idx]
            img_nzzf_mask = mask_nzzf[idx]
            
            merge_mask = np.zeros_like(img_ydj_mask, dtype=np.uint8)

            if cat == 0:
                merge_mask[img_ydj_mask==1] = 1
            elif cat == 1:
                merge_mask[img_otherggj_mask==1] = 1
            elif cat == 2:
                merge_mask[img_pxzf_mask==1] = 1
            elif cat == 3:
                merge_mask[img_nzzf_mask==1] = 1
            elif cat == 4:
                merge_mask[img_ydj_mask==1] = 1
                merge_mask[img_otherggj_mask==1] = 2
                merge_mask[img_pxzf_mask==1] = 3
                merge_mask[img_nzzf_mask==1] = 4
            
            data_mask_list.append(merge_mask)
        
        array_dicom = np.array(data_mask_list, dtype=np.uint8)
        # NOTE 由鉴影生成的mask方向与工程排序方式不一致，需要严格注意
        
        mask_sitk_img = SimpleITK.GetImageFromArray(array_dicom, isVector=False)
        scan_sitk_img = SimpleITK.ReadImage(os.path.join(image_folder, patient_name + '.mha'))
        mask_sitk_img.CopyInformation(scan_sitk_img)
        
        if mask_sitk_img.GetSize() != scan_sitk_img.GetSize():
            mask_sitk_img = sitk_resample_to_image(
                mask_sitk_img, 
                scan_sitk_img, 
                interpolator=SimpleITK.sitkNearestNeighbor, 
                output_pixel_type=SimpleITK.sitkUInt8)
        
        SimpleITK.WriteImage(mask_sitk_img, 
                             os.path.join(dst_folder, patient_name + ".mha"), 
                             useCompression=True)
        return True

    except Exception as e:
        return e


def get_mask_3d(src_folder, dst_folder, image_folder, cat):
    os.makedirs(dst_folder,  exist_ok=True)
    with Pool(32) as p:
        results = []
        failed = []
        for slide_item in os.listdir(src_folder):
            if os.path.exists(os.path.join(dst_folder, slide_item + '.mha')):
                continue
            
            task = p.apply_async(process_mask_one_case, (src_folder, dst_folder, image_folder, cat, slide_item))
            results.append(task)
        
        for i, result in tqdm(enumerate(results), dynamic_ncols=True, total=len(results), desc='Processing'):
            success = result.get()
            if not success is True:
                failed.append({
                    '序列编号': slide_item,
                    '失败原因': str(success),
                })
        
        json.dump(failed, open(os.path.join(dst_folder, 'failed.json'), 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    '''
    用于转换来自鉴影标注的Label-Split的MHA文件
    将其合并为一个mha文件，标注将转换为label-index模式。
        
    Args
        - src_folder: 鉴影标注的mha文件夹路径
        - dst_folder: 转换后的mha文件夹路径
        - image_folder: 对应的扫描序列文件夹路径，用以对齐image和label的元信息。
        
    NOTE 鉴影标注的顺序遵循Z轴方向降序排列，需要注意与工程的Dicom转换方向是否一致
        必要时建议可视化观察。
    '''
    
    src_folder = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Batch5_7986/raw/label'
    dst_folder = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Batch5_7986/mha_original_EngineerSort/label'
    image_folder = '/fileser51/zhangyiqin.sx/Sarcopenia_Data/Batch5_7986/mha_original_EngineerSort/image'

    cat = 4 # all

    # get_mask_2d(src_folder, dst_folder, cat, l3_csv)
    get_mask_3d(src_folder, dst_folder, image_folder, cat)

