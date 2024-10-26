"""
MGAM Datatoolkits Totalsegmentator 3D Pre-Crop Script.

The source structure:

data_root
├── case1/
│   ├── ct.mha
│   └── segmentations.mha
│
├── case2/
│   ├── ct.mha
│   └── segmentations.mha
│
└── ...

dest_root
├── case1/
│   ├── case1_0.npz
│   │   ├── img
│   │   └── gt_seg_map
│   │
│   ├── case1_1.npz
│   │   └── ...
│   │
│   └── ...
│
├── case2/
│   └── ...
│
└── ...

"""


import os
import argparse
import json
import multiprocessing as mp
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk



def crop_per_series(args:tuple):
    cropper, series_path, num_cropped, save_folder = args
    cropper: RandomCrop3D
    series_path: str
    num_cropped: int
    save_folder: str
    
    image_itk_image = sitk.ReadImage(os.path.join(series_path, 'ct.mha'))
    anno_itk_image = sitk.ReadImage(os.path.join(series_path, 'segmentations.mha'))
    image_array = sitk.GetArrayFromImage(image_itk_image)
    anno_array = sitk.GetArrayFromImage(anno_itk_image)
    data = {
        'img': image_array,
        'gt_seg_map': anno_array,
        'seg_fields': ['gt_seg_map'],
    }
    
    if num_cropped is None:
        num_cropped = int(np.prod(np.array(image_array.shape) // np.array(cropper.crop_size)))
    
    os.makedirs(save_folder, exist_ok=True)
    for crop_idx in range(num_cropped):
        save_path = os.path.join(save_folder, f'{os.path.basename(save_folder)}_{crop_idx}.npz')
        if os.path.exists(save_path):
            continue
        
        crop_bbox = cropper.crop_bbox(data)
        cropped_image_array = cropper.crop(image_array, crop_bbox).astype(np.int16)
        cropped_anno_array = cropper.crop(anno_array, crop_bbox).astype(np.uint8)
        np.savez_compressed(
            file=save_path,
            img=cropped_image_array,
            gt_seg_map=cropped_anno_array)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Pre-Random-Crop 3D')
    argparser.add_argument('source_mha_folder', type=str,   help='The folder containing mha files.')
    argparser.add_argument('dest_npz_folder',   type=str,   help='The folder to save npz files.')
    argparser.add_argument('--crop-size',       type=int,   nargs=3, required=True, help='The size of cropped volume.')
    argparser.add_argument('--crop-cat-max',    type=float, default=0.9, help='Max ratio for single catagory can occupy.')
    argparser.add_argument('--num-cropped',     type=int,   default=None, help='The number of cropped volumes per series.')
    argparser.add_argument('--ignore-index',    type=int,   default=255, 
                           help='The index to ignore in segmentation. '
                                'It will not taken into consideration during '
                                'the determination of whether the cropped patch '
                                'meets the `crop-cat-max` setting.')
    argparser.add_argument('--mp', action='store_true',     default=False, help='Whether to use multiprocessing.')
    args = argparser.parse_args()
    
    os.makedirs(args.dest_npz_folder, exist_ok=True)
    json.dump(vars(args),
              open(os.path.join(args.dest_npz_folder, 'crop_meta.json'), 'w'),
              indent=4)
    
    from mgamdata.mm.mmseg_Dev3D import RandomCrop3D
    task_list = []
    for series in os.listdir(args.source_mha_folder):
        task_list.append((
            RandomCrop3D(args.crop_size, args.crop_cat_max, args.ignore_index),
            os.path.join(args.source_mha_folder, series),
            args.num_cropped,
            os.path.join(args.dest_npz_folder, series)))
    
    results = []
    if args.mp:
        with mp.Pool() as pool:
            fetcher = pool.imap_unordered(crop_per_series, task_list)
            for result in tqdm(fetcher, "Cropping", total=len(task_list), 
                               dynamic_ncols=True, leave=False):
                results.append(result)
    else:
        for task in tqdm(task_list, "Cropping",
                         dynamic_ncols=True, leave=False):
            result = crop_per_series(task)
            results.append(result)
    
    print(f"Finished cropping {len(results)} series.")
