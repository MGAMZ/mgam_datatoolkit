import os
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
import pandas as pd

from mgamdata.dataset.Totalsegmentator.meta import (
    META_CSV_PATH, CLASS_INDEX_MAP)



class meta_file_handler:
    """ Original columns in meta csv file:
        image_id
        age
        gender
        institute
        study_type
        split
        manufacturer
        scanner_model
        kvp	pathology
        pathology_location
    """
    def __init__(self, meta_csv_path:str):
        self.meta_csv_path = meta_csv_path
        self.meta_df = pd.read_csv(meta_csv_path)
        self._init_dfs()
    
    def _init_dfs(self):
        df_template = self.meta_df.copy(deep=True)
        for feature in ['mean', 'std', 'min', 'max', 'lower_bound', 'upper_bound']:
            df_template[feature] = pd.Series(dtype=np.float32)
        
        self.whole_df = df_template.copy(deep=True)
        self.per_class_dfs = {class_name: df_template.copy(deep=True)
                              for class_name in CLASS_INDEX_MAP.keys()}
        
        for class_name in CLASS_INDEX_MAP.keys():
            self.whole_df["mean_"+class_name] = pd.Series(dtype=np.float32)
        self.whole_df = self.whole_df.copy()
    
    def register(self, class_name, series_id, **kwargs):
        if class_name == 'whole':
            for feature_name, value in kwargs.items():
                self.whole_df.loc[self.whole_df['image_id'] == series_id, feature_name] = value
        
        elif class_name in CLASS_INDEX_MAP.keys():
            for feature_name, value in kwargs.items():
                self.per_class_dfs[class_name].loc[
                        self.per_class_dfs[class_name]['image_id'] == series_id, feature_name
                    ] = value
                if feature_name == "mean":
                    self.whole_df.loc[self.whole_df['image_id'] == series_id, 
                                      "mean_"+class_name
                        ] = value
        
        else:
            raise ValueError(f'Invalid class name: {class_name}')
    
    def save(self):
        save_folder = os.path.join(os.path.dirname(self.meta_csv_path), 'distribution')
        os.makedirs(save_folder, exist_ok=True)
        
        whole_distribution_path = os.path.join(save_folder, 'whole_distribution.csv')
        self.whole_df.to_csv(whole_distribution_path, index=False)
        
        for class_name, df in self.per_class_dfs.items():
            class_distribution_path = os.path.join(save_folder, f'{class_name}_distribution.csv')
            df.to_csv(class_distribution_path, index=False)


def parse_one_case(case_folder:str):
    itk_ct = sitk.ReadImage(os.path.join(case_folder, 'ct.mha'))
    scan_array = sitk.GetArrayFromImage(itk_ct).astype(np.int16)
    itk_anno = sitk.ReadImage(os.path.join(case_folder, 'segmentations.mha'))
    anno_array = sitk.GetArrayFromImage(itk_anno).astype(np.uint8)

    # whole distribution
    distribution = {
        'series_id': os.path.basename(case_folder),
        'whole': {
            'mean': np.mean(scan_array).astype(np.float32),
            'std': np.std(scan_array).astype(np.float32),
            'min': np.min(scan_array).astype(np.int32),
            'max': np.max(scan_array).astype(np.int32),
            'lower_bound': np.percentile(scan_array, 5).astype(np.int32),
            'upper_bound': np.percentile(scan_array, 95).astype(np.int32),
        }
    }

    # per class distribution
    class_idxs = np.unique(anno_array)
    class_names = [name for name, idx in CLASS_INDEX_MAP.items() if idx in class_idxs]
    for class_name in class_names:
        class_idx = CLASS_INDEX_MAP[class_name]
        class_mask = anno_array == class_idx
        distribution[class_name] = {
            'mean': np.mean(scan_array[class_mask]).astype(np.float32),
            'std': np.std(scan_array[class_mask]).astype(np.float32),
            'min': np.min(scan_array[class_mask]).astype(np.int32),
            'max': np.max(scan_array[class_mask]).astype(np.int32),
            'lower_bound': np.percentile(scan_array[class_mask], 5).astype(np.int32),
            'upper_bound': np.percentile(scan_array[class_mask], 95).astype(np.int32)
        }
        
    return distribution


def parse_all_cases(data_root:str, meta_csv_path:str, mp:bool=False):
    cases = []
    task_list = [os.path.join(data_root, case_folder) 
                 for case_folder in os.listdir(data_root) 
                 if os.path.isdir(os.path.join(data_root, case_folder))]
    meta_writer = meta_file_handler(meta_csv_path)
    
    if mp:
        with Pool() as p:
            fetcher = p.imap_unordered(parse_one_case, task_list, chunksize=16)
            for case in tqdm(fetcher, 
                             total=len(task_list),
                             desc="Parsing",
                             leave=False,
                             dynamic_ncols=True):
                cases.append(case)
                
                series_id = case.pop('series_id')
                for class_name, class_distribution in case.items():
                    meta_writer.register(class_name, series_id, **class_distribution)
        meta_writer.save()
    
    else:
        for case_folder in tqdm(task_list, 
                                leave=False, 
                                dynamic_ncols=True):
            case = parse_one_case(case_folder)
            cases.append(case)
            
            series_id = case.pop('series_id')
            for class_name, class_distribution in case.items():
                meta_writer.register(class_name, series_id, **class_distribution)
        meta_writer.save()
    
    return cases



if __name__ == '__main__':
    parse_all_cases(
        data_root='', 
        meta_csv_path='',
        mp=True
    )
