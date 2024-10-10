import argparse
import os
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path

import SimpleITK as sitk

from mgamdata.process.kmeans import sarcopenia_muscle_subsegmentation_fromITK
from mgamdata.utils.search_tool import search_mha_file
from mgamdata.dataset.RenJi_Sarcopenia import GT_FOLDERS_PRIORITY_ORIGINAL_ENGINEERSORT


def process_one_pair(image_path:str, mask_path:str, output_folder:str):
    output_path = os.path.join(args.output_folder, os.path.basename(mask_path))
    if os.path.exists(output_path) or image_path is None:
        return 1
    post_seg_mask = sarcopenia_muscle_subsegmentation_fromITK(image_path, mask_path)
    sitk.WriteImage(post_seg_mask, 
                    os.path.join(output_folder, os.path.basename(image_path)), 
                    useCompression=True)
    return 0



def parse_args():
    parser = argparse.ArgumentParser("肌少症Kmeans后处理，对肌肉进行细分，得出肌间脂肪类。")
    parser.add_argument("pred_folder", type=str, help="标注序列ITK MHA文件夹")
    parser.add_argument("output_folder", type=str, help="输出文件夹")
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    
    pred_files = [os.path.join(args.pred_folder, i) for i in os.listdir(args.pred_folder) if i.endswith('.mha')]
    image_files = [search_mha_file(GT_FOLDERS_PRIORITY_ORIGINAL_ENGINEERSORT, seriesUID, 'image') 
                    for seriesUID in [Path(file).stem for file in pred_files if file.endswith('.mha')]]
    assert len(pred_files) == len(image_files)
    os.makedirs(args.output_folder, exist_ok=True)
    
    with mp.Pool(24) as p:
        results = []
        for image_path, mask_path in zip(image_files, pred_files):
            result = p.apply_async(process_one_pair, args=(image_path, mask_path, args.output_folder))
            results.append(result)
    
        for result in tqdm(results,
                           desc="Post Segment",
                           dynamic_ncols=True,
                           leave=False,
                           mininterval=1,
                           total=len(pred_files)):
            result.get()
            
