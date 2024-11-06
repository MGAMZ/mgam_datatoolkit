from abc import abstractmethod
import os
import argparse
import json
import multiprocessing as mp
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk

from mgamdata.mm.mmseg_Dev3D import RandomCrop3D


class PreCropper3D:
    def __init__(self):
        self.main()
    
    def arg_parse(self):
        argparser = argparse.ArgumentParser('Pre-Random-Crop 3D')
        argparser.add_argument('source_mha_folder', type=str,   help='The folder containing mha files.')
        argparser.add_argument('dest_npz_folder',   type=str,   help='The folder to save npz files.')
        argparser.add_argument('--crop-size',       type=int,   nargs=3,        required=True, help='The size of cropped volume.')
        argparser.add_argument('--crop-cat-max',    type=float, default=1.,     help='Max ratio for single catagory can occupy.')
        argparser.add_argument('--num-cropped',     type=int,   default=None,   help='The number of cropped volumes per series.')
        argparser.add_argument('--ensure-index',    type=int,   default=None,   nargs='+', 
                               help='The index to ensure in segmentation.')
        argparser.add_argument('--ensure-ratio',    type=float, default=None,    help='The chance for an ensurance to perform.')
        argparser.add_argument('--ignore-index',    type=int,   default=255, 
                               help='The index to ignore in segmentation. '
                                    'It will not taken into consideration during '
                                    'the determination of whether the cropped patch '
                                    'meets the `crop-cat-max` setting.')
        argparser.add_argument('--mp', action='store_true',     default=False, help='Whether to use multiprocessing.')
        self.args = argparser.parse_args()
    
    @abstractmethod
    def parse_task(self) -> list[tuple[RandomCrop3D, str, str, int, str]]:
        """
        Task List, each task contains:
            - RandomCrop3D Class
            - image_itk_path
            - anno_itk_path
            - save_folder
        """
        ...
    
    def main(self):
        self.arg_parse()
        os.makedirs(self.args.dest_npz_folder, exist_ok=True)
        json.dump(vars(self.args),
                  open(os.path.join(self.args.dest_npz_folder, 'crop_meta.json'), 'w'),
                  indent=4)
        self.task_list = self.parse_task()
        
        results = []
        if self.args.mp:
            with mp.Pool() as pool:
                fetcher = pool.imap_unordered(self.crop_per_series, self.task_list)
                for result in tqdm(fetcher, "Cropping", total=len(self.task_list), 
                                dynamic_ncols=True, leave=False):
                    results.append(result)
        else:
            for task in tqdm(self.task_list, "Cropping",
                            dynamic_ncols=True, leave=False):
                result = self.crop_per_series(task)
                results.append(result)
        
        print(f"Finished cropping {len(results)} series.")
    
    def all_index_ensured(self, label:np.ndarray):
        if self.args.ensure_index is None or np.random.rand() > self.args.ensure_ratio:
            return True
        else:
            return any(index not in label for index in self.args.ensure_index)
    
    def crop_per_series(self, args:tuple):
        cropper, image_itk_path, anno_itk_path, save_folder = args
        cropper: RandomCrop3D
        os.makedirs(save_folder, exist_ok=True)
        
        for crop_idx, (img_array, anno_array) in enumerate(
            self.Crop3D(cropper, image_itk_path, anno_itk_path)
        ):
            save_path = os.path.join(
                save_folder, 
                f'{os.path.basename(save_folder)}_{crop_idx}.npz')
            np.savez_compressed(
                file=save_path,
                img=img_array,
                gt_seg_map=anno_array)
            

    def Crop3D(self,
               cropper, # type: ignore
               image_itk_path:str, 
               anno_itk_path:str):
        from mgamdata.mm.mmseg_Dev3D import RandomCrop3D
        cropper: RandomCrop3D
        
        image_itk_image = sitk.ReadImage(image_itk_path)
        anno_itk_image = sitk.ReadImage(anno_itk_path)
        image_array = sitk.GetArrayFromImage(image_itk_image)
        anno_array = sitk.GetArrayFromImage(anno_itk_image)
        data = {
            'img': image_array,
            'gt_seg_map': anno_array,
            'seg_fields': ['gt_seg_map'],
        }
        
        if self.args.num_cropped is None:
            num_cropped = int(np.prod(np.array(image_array.shape) // np.array(cropper.crop_size)))
        else:
            num_cropped = self.args.num_cropped
        
        for i in range(num_cropped):
            crop_bbox = cropper.crop_bbox(data)
            cropped_anno_array:np.ndarray = cropper.crop(
                anno_array, crop_bbox).astype(np.uint8)
            
            if self.all_index_ensured(cropped_anno_array):
                cropped_image_array:np.ndarray = cropper.crop(
                    image_array, crop_bbox).astype(np.int16)
                yield cropped_image_array, cropped_anno_array
            
            else:
                tqdm.write(f"deprecated due to failing to ensure index: {anno_itk_path} | crop_idx: {i}")
