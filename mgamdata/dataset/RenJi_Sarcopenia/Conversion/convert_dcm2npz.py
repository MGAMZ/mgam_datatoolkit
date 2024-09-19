import os
import os.path as osp
import warnings
from multiprocessing import Pool
from colorama import Fore, Style

from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

from aitrox.utils.sitk_toolkit import LoadDcmAsSitkImage, LoadMhaAnno



"""
    This script is designed to convert the original data format to numpy format.
    Input:
        - Dcm Files Scan
        - Mha Files Scan, the number of mha is the number of the class, with class_idx being the path's shuffix.
    Output:
        - Npz File with keys:
            - img                   (H, W, D)
            - mask_class_idx        (H, W, D)
            - mask_class_channel    (C, H, W, D)
"""

def ConvertDcmMhaPair_to_Npy(dcm_root, mha_root, patient, out_root, spacing):
    try:
        sitk_image, ori_spacing, original_size, resampled_size = LoadDcmAsSitkImage(
            osp.join(dcm_root, patient), spacing)
        tqdm.write(f"{patient[-10:]} original spacing {ori_spacing}, output size {resampled_size}")
    except Exception as e:
        warnings.warn(f"Corrupted Sample: {patient}.\n Reason: {e}.", UserWarning)
        return -1
    
    anno_with_class_channel, anno_without_class_channel = LoadMhaAnno(
        mha_root, patient, ori_spacing, spacing, resampled_size)
    
    if anno_with_class_channel is None or anno_without_class_channel is None:
        file_name = osp.join(out_root, 'NOMASK_'+patient+'.npz')
    else:
        tqdm.write(f"{patient[-10:]} mask shape {anno_with_class_channel.shape}")
        file_name = osp.join(out_root, patient+'.npz')
    
    np.savez_compressed(file = file_name, 
                        img = sitk.GetArrayFromImage(sitk_image), 
                        mask_class_idx = anno_without_class_channel, 
                        mask_class_channel = anno_with_class_channel,
                        original_spacing = np.array(ori_spacing)[::-1],
                        original_size = np.array(original_size)[::-1],
                        current_spacing = np.array(spacing)[::-1],
                        current_size = np.array(resampled_size)[::-1],
                    )
    tqdm.write(f"Saved {osp.basename(file_name)}")


def process_one_patient(original_img_root, original_mask_root, 
                        out_root, spacing, patient):
    os.makedirs(out_root, exist_ok=True)
    if osp.exists(osp.join(out_root, patient+'.npz')) or \
       osp.exists(osp.join(out_root, 'NOMASK_'+patient+'.npz')) or \
       osp.exists(osp.join(out_root, 'CORRUPTED_'+patient+'.npz')):
        return
    else:
        ConvertDcmMhaPair_to_Npy(original_img_root, original_mask_root, patient, out_root, spacing)


def convert(batch_data, spacing:tuple):
    for idx, (original_img_root, original_mask_root, out_root) in enumerate(batch_data):
        patient_instance = os.listdir(original_img_root)
        with Pool(8) as p:
            results = []
            for patient in patient_instance:
                result = p.apply_async(
                    process_one_patient, 
                    args=(original_img_root, original_mask_root, 
                        out_root, spacing, patient))
                results.append(result)
            
            for result in tqdm(results, desc='MultiProcessing'):
                result.get()


def test_file(path):
    try:
        zip = np.load(path, allow_pickle=True)
        img = zip['img']
        mask_class_idx = zip['mask_class_idx']
        mask_class_channel = zip['mask_class_channel']
        assert img.ndim == 3, f"Invalid Image Dimension: {img.shape}"
        if not osp.basename(path).startswith('NOMASK'):
            assert mask_class_idx.ndim == 3, f"Invalid Mask Class Index Dimension: {mask_class_idx.shape}"
            assert mask_class_channel.ndim == 4, f"Invalid Mask Class Channel Dimension: {mask_class_channel.shape}"
        return None

    except Exception as e:
        if osp.basename(path).startswith('CORRUPTED_'):
            return None
        file_name = "CORRUPTED_" + osp.basename(path)
        os.rename(path, osp.join(osp.dirname(path), file_name))
        return f"Corrupted File Labeled as CORRPUTED: {path}.\nReason: {e}."


def test(batch_data):
    with Pool(24) as p:
        npzs = []
        for idx, (pic_root, mask_root, out_root) in enumerate(batch_data):
            for roots, dirs, files in os.walk(out_root):
                for file in files:
                    npzs.append(osp.join(roots, file))

        result_iter = p.imap_unordered(test_file, npzs, chunksize=4)
        for check in tqdm(result_iter, desc='Testing', total=len(npzs)):
            if check is not None:
                tqdm.write(check)



if __name__ == '__main__':
    target_spacing = [1, None, None] # D S W
    
    # List of List:
    # [original_img_root(dcm), original_mask_root(mha), out_root(npz)]
    # SEE TOP ABOVE.
    batch_data = [
        ['/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/first_try/pic_',
         '/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/first_try/mask_',
         f'/fileser51/zhangyiqin.sx/Sarcopenia_Data/Spacing{target_spacing}/Batch1'],
        
        ['/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/second_try/pic_',
         '/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/second_try/mask_',
         f'/fileser51/zhangyiqin.sx/Sarcopenia_Data/Spacing{target_spacing}/Batch2'],
        
        ['/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/third_try/pic_',
         '/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/third_try/mask_',
         f'/fileser51/zhangyiqin.sx/Sarcopenia_Data/Spacing{target_spacing}/Batch3'],
        
        ['/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/fourth_try/pic_',
         '/fileser51/zhangwh.lw/workspace/projects/renji_sarcopenia/data/fourth_try/mask_',
         f'/fileser51/zhangyiqin.sx/Sarcopenia_Data/Spacing{target_spacing}/Batch4'],
    ]
    try:
        convert(batch_data, spacing=[target_spacing[1], 
                                     target_spacing[2], 
                                     target_spacing[0]])
    except Exception as e:
        print(e)
        print(Fore.RED, 
              'WARNING: Convert Interrupted. There may exist some corrupted npz, for their incomplete writing.',
              'An additional test is needed to check the integrity of all npz files.',
              Style.RESET_ALL)
        exit(-1)
        
    test(batch_data)
    
        
        

