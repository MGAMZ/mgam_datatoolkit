import json
import os
import pdb
import random
import re
import shutil
from concurrent import futures
from concurrent.futures import Future
from multiprocessing import Pool, cpu_count
from textwrap import indent
from typing import Dict, Iterable, List, Sequence, Union

import cv2
import numpy as np
import orjson
from genericpath import isfile
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from rich.progress import Progress
from tqdm import tqdm


class SA_Med2D:
    MODALITIES = ['ct_00', 'ct_cbf', 'ct_cbv', 'ct_mtt', 'ct_tmax', 'dermoscopy_00', 
               'endoscopy_00', 'fundus_photography', 'mr_00', 'mr_adc', 'mr_cbf', 'mr_cbv', 
               'mr_cmr', 'mr_dwi', 'mr_flair', 'mr_hbv', 'mr_lge', 'mr_mprage', 'mr_mtt', 
               'mr_pd', 'mr_rcbf', 'mr_rcbv', 'mr_t1', 'mr_t1c', 'mr_t1ce', 'mr_t1gd', 
               'mr_t1w', 'mr_t2', 'mr_t2w', 'mr_tmax', 'mr_ttp', 'pet_00', 'ultrasound_00', 'x_ray']	# NUM:34
    IMAGE_NAME_DEFINITION = ['modality_sub-modality', 'dataset name', 'ori name', 'slice_direction', 'slice_index']
    MASK_NAME_DEFINITION = IMAGE_NAME_DEFINITION + ['class_instance', 'id']
    NAME_POS_MAP = {name: i for i, name in enumerate(MASK_NAME_DEFINITION)}
    CLASS_ID_OFFSET = 1

    @classmethod
    def analyze_file_name(cls, file_name:str) -> Dict[str, str]:
        part = file_name.rstrip('.png').split('--')
        assert part[0] in cls.MODALITIES
        if len(part) == 4:
            file_type = 'image'
            slice_direction, slice_index = part[-1].split('_')
            part = part[:-1] + [slice_direction, slice_index]
        elif len(part) == 5:
            file_type = 'mask'
            slice_direction, slice_index = part[-2].split('_')
            class_instance, idx = part[-1].split('_')
            part = part[:-2] + [slice_direction, slice_index, class_instance, idx]
        else:
            raise ValueError(f'Found {len(part)} name parts in file name {file_name}, expect{len(cls.IMAGE_NAME_DEFINITION)} or {len(cls.MASK_NAME_DEFINITION)}')

        analyze_result = {'type': file_type}
        for name, value in zip(cls.IMAGE_NAME_DEFINITION if file_type == 'image' else cls.MASK_NAME_DEFINITION, 
                               part):
            analyze_result[name] = value
        
        return analyze_result

    @classmethod
    def instance_to_segmentation(cls, instances:List[np.ndarray]) -> np.ndarray:
        # instance: (class_id, ndarray)  or the Future of it
        if isinstance(instances[0][1], futures.Future):
            example = instances[0][1].result()
        else:
            example = instances[0][1]
        segmentation_map = np.zeros(example.shape, dtype=np.uint16)
        for class_id, instance in instances:
            if isinstance(instance, futures.Future):
                instance = instance.result()
            assert instance.shape == segmentation_map.shape[:2]
            segmentation_map[instance > 0] = int(class_id) + cls.CLASS_ID_OFFSET
        return segmentation_map	# (h, w)


class DataStructurer(SA_Med2D):
    def __init__(self, src_root:str, dest_root:str, mode:str):
        self.src_root = src_root
        self.dest_root = dest_root
        self.ModalityRootAfterInit = os.path.join(dest_root, 'CaseSeperated')
        self.image_path = os.path.join(self.src_root, 'images')
        self.mask_path = os.path.join(self.src_root, 'masks')

        if mode == 'init':
            print('正在执行init')
            # self.Hierarchical_Folder_init() # 建立多级文件夹存放png图像
            # self.Split_ImageMaskMapJSON_into_DatasetSeperated() # 分割Image-Mask映射JSON，分级到dataset层面
            # self.Create_CaseSlice_Map_for_Each_Dataset() # 统计每个数据集的每个case的slice数量，分dataset存储为JSON
            # self.Calculate_Dataset_Value_Distributions()
            # self.Split_ClassLabelMap_into_ModalitySeperated()
            # self.Create_Dataset_NumSamples_Map_for_Each_Dataset()
            # self.Filter_ClassLabelMap_to_Truely_Existed_with_ModalitySeperated()
        else:
            raise NotImplementedError
    
    @staticmethod
    def png_to_ndarray(path:str) -> np.ndarray:
        return np.array(Image.open(path))
    
    @staticmethod
    def stack_and_compress_to_npz(path, data:Sequence[ArrayLike]) -> None:
        stacked_data = np.stack(data, axis=0)
        np.savez_compressed(path, data=stacked_data)
        return None

    @staticmethod
    def calculate_distributions_for_one_slice(path):
        png_file = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # (H, W, C)
        if len(png_file.shape) == 2:
            png_file = png_file[:, :, np.newaxis]
        arr = png_file.reshape(png_file.shape[0]*png_file.shape[1], png_file.shape[2])
        # 统计三通道参数
        return [np.min(arr,axis=0), np.max(arr,axis=0), np.mean(arr,axis=0), np.std(arr,axis=0)]

    @staticmethod
    def calculate_label_distributions_for_one_slice(path):
        png_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # (H, W, C)
        # 统计出现的label index
        return np.unique(png_array)


    def Split_ImageMaskMapJSON_into_DatasetSeperated(self):
        dict_cache = {}
        ImageMaskMapJSONPath = os.path.join(self.src_root, 'SAMed2D_v1.json')
        source_map = orjson.loads(open(ImageMaskMapJSONPath, 'r').read())

        for key, value in tqdm(source_map.items(), total=len(source_map)):
            name_components = SA_Med2D.analyze_file_name(key.split('/')[-1])
            modality_dict = dict_cache.get(name_components['modality_sub-modality'], {})
            dataset_dict = modality_dict.get(name_components['dataset name'], {})
            dataset_dict[key] = value
            modality_dict[name_components['dataset name']] = dataset_dict
            dict_cache[name_components['modality_sub-modality']] = modality_dict
        
        for modality, modality_dict in dict_cache.items():
            for dataset_name, dataset_dict in modality_dict.items():
                json.dump(dataset_dict, open(
                        os.path.join(self.ModalityRootAfterInit, modality, dataset_name, f'{dataset_name}_img_mask_map.json'), 'w'
                    ), indent=4)


    def Split_ClassLabelMap_into_ModalitySeperated(self):
        ClassLabelMapJSONPath = os.path.join(self.src_root, 'SAMed2D_v1_class_mapping_id.json')
        source_map = orjson.loads(open(ClassLabelMapJSONPath, 'r').read())
        
        for modality in os.listdir(self.ModalityRootAfterInit):
            if os.path.isfile(os.path.join(self.ModalityRootAfterInit, modality)): continue
            modality_class_map = {}
            for dataset in os.listdir(os.path.join(self.ModalityRootAfterInit, modality)):
                if os.path.isfile(os.path.join(self.ModalityRootAfterInit, modality, dataset)): continue
                dataset_class_map = source_map[dataset]
                modality_class_map[dataset] = dataset_class_map
            
            json.dump(modality_class_map, open(
                    os.path.join(self.ModalityRootAfterInit, modality, f'{modality}_class_map.json'), 'w'
                ), indent=4)

    @classmethod
    def GetUniqueLabelIndex_FromPNG(cls, png_path:str) -> List[int]:
            return np.unique(cls.png_to_ndarray(png_path))


    def Filter_ClassLabelMap_to_Truely_Existed_with_ModalitySeperated(self):
        p = Pool(cpu_count())
        
        for modality in os.listdir(self.ModalityRootAfterInit):
            if os.path.isfile(os.path.join(self.ModalityRootAfterInit, modality)): continue
            modality_class_map = json.load(open(os.path.join(
                self.ModalityRootAfterInit, modality, f'{modality}_class_map.json'), 'r'))
            
            for dataset in os.listdir(os.path.join(self.ModalityRootAfterInit, modality)):
                dataset_class_map_path = os.path.join(self.ModalityRootAfterInit, modality, dataset, f'{dataset}_exist_class_map.json')
                if os.path.exists(dataset_class_map_path):
                    modality_class_map[dataset] = json.load(open(dataset_class_map_path, 'r'))
                    continue
                if os.path.isfile(os.path.join(self.ModalityRootAfterInit, modality, dataset)):
                    continue
                
                dataset_class_map = modality_class_map[dataset]
                dataset_exist_class_index = []
                task_list = []
                
                for roots, dirs, files in os.walk(os.path.join(self.ModalityRootAfterInit, modality, dataset)):
                    for file in files:
                        if file.startswith('mask_'):
                            task_list.append(os.path.join(roots, file))
                
                results = p.imap_unordered(self.GetUniqueLabelIndex_FromPNG, task_list, chunksize=20)
                
                for unique_index_per_mask in tqdm(results, desc=f'{modality}_{dataset}', total=len(task_list)):
                    for index in unique_index_per_mask:
                        if int(index) in dataset_exist_class_index: continue
                        else: dataset_exist_class_index.append(int(index))
                
                dataset_exist_class_map = {'background': 0}
                dataset_exist_class_index = sorted(dataset_exist_class_index)
                for class_name, class_index in dataset_class_map.items():
                    if int(class_index)+1 in dataset_exist_class_index:
                        dataset_exist_class_map[class_name] = int(class_index) + 1
                
                modality_class_map[dataset] = dataset_exist_class_map
                
                json.dump(dataset_exist_class_map, open(dataset_class_map_path, 'w'), indent=4)

            json.dump(modality_class_map, open(
                    os.path.join(self.ModalityRootAfterInit, modality, f'{modality}_exist_class_map.json'), 'w'
                ), indent=4)


    def Create_CaseSlice_Map_for_Each_Dataset(self):
        def slice_name_to_slice_index(slice_name:Union[str, List]) -> Union[str, List]:
            if isinstance(slice_name, List): 
                for i, name in enumerate(slice_name):
                    slice_name[i] = slice_name_to_slice_index(name)
                return slice_name
            elif isinstance(slice_name, str):
                return slice_name.lstrip('image_').lstrip('mask_').rstrip('.png')
            else:
                raise TypeError(f'{type(slice_name)} is not supported')
        
        for modality in tqdm(os.listdir(self.ModalityRootAfterInit), desc='Modality'):
            dataset_root = os.path.join(self.ModalityRootAfterInit, modality)
            if not os.path.isdir(dataset_root): continue
            
            for dataset in tqdm(os.listdir(dataset_root), desc=modality):
                case_root = os.path.join(dataset_root, dataset)
                if not os.path.isdir(case_root): continue
                dataset_slice_map = {}
                
                for Case in os.listdir(case_root):
                    direction_root = os.path.join(case_root,Case)
                    if not os.path.isdir(direction_root): continue
                    
                    case_slice_map = {}
                    for direction in os.listdir(direction_root):
                        slice_root = os.path.join(direction_root, direction)
                        if not os.path.isdir(slice_root): continue
                        
                        slice_name_of_one_case = []
                        for slice in os.listdir(slice_root):
                            if slice.endswith('.png') and slice.startswith('image_'): 
                                slice_name_of_one_case.append(slice.rstrip('.png'))

                        case_slice_map[direction] = slice_name_to_slice_index(slice_name_of_one_case)
                    dataset_slice_map[Case] = case_slice_map
                
                json.dump(dataset_slice_map, 
                          open(os.path.join(case_root, f'{dataset}_CaseSlice_map.json'), 'w'), 
                          indent=4)


    def Create_Dataset_NumSamples_Map_for_Each_Dataset(self):
        modality_num_slice_map = {'all': {}}
        for modality in tqdm(os.listdir(self.ModalityRootAfterInit), desc='Modality'):
            dataset_root = os.path.join(self.ModalityRootAfterInit, modality)
            if not os.path.isdir(dataset_root): continue
            
            dataset_num_slice_map = {}
            for dataset in tqdm(os.listdir(dataset_root), desc=modality):
                case_root = os.path.join(dataset_root, dataset)
                if not os.path.isdir(case_root): continue
                count = 0
                
                for Case in os.listdir(case_root):
                    direction_root = os.path.join(case_root, Case)
                    if not os.path.isdir(direction_root): continue

                    for direction in os.listdir(direction_root):
                        slice_root = os.path.join(direction_root, direction)
                        if not os.path.isdir(slice_root): continue
                        
                        for slice in os.listdir(slice_root):
                            if slice.endswith('.png') and slice.startswith('image_'): 
                                count += 1
                
                dataset_num_slice_map[dataset] = count
            
            modality_num_slice_map['all'][modality] = sum([count for dataset, count in dataset_num_slice_map.items()])
            modality_num_slice_map[modality] = dataset_num_slice_map
            json.dump(dataset_num_slice_map, 
                    open(os.path.join(dataset_root, f'{modality}_NumSlice_map.json'), 'w'), 
                    indent=4)
        json.dump(modality_num_slice_map, 
                    open(os.path.join(self.ModalityRootAfterInit, f'NumSlice_map.json'), 'w'), 
                    indent=4)


    def Create_AtomClassLabelMap(self):
        # This function relies on the map created by method 
        # 'Filter_ClassLabelMap_to_Truely_Existed_with_ModalitySeperated', 
        # so please run the above method first.
        for modality in os.listdir(self.ModalityRootAfterInit):
            if os.path.isfile(os.path.join(self.ModalityRootAfterInit, modality)): continue
            for dataset in os.listdir(os.path.join(self.ModalityRootAfterInit, modality)):
                atom_class_map_path = os.path.join(self.ModalityRootAfterInit, modality, dataset, f'{dataset}_atom_class_map.json')
                if os.path.exists(atom_class_map_path) or \
                   os.path.isfile(os.path.join(
                       self.ModalityRootAfterInit, modality, dataset)):
                    continue
                
                dataset_class_map = json.load(open(os.path.join(
                    self.ModalityRootAfterInit, modality, dataset, 
                    f'{dataset}_exist_class_map.json'), 'r'))
                atom_classes_dict = {
                    class_name: dataset_class_map[class_name]
                    for class_name in dataset_class_map.keys()
                    if not class_name.startswith('union_')
                }
                
                json.dump(atom_classes_dict, 
                          open(atom_class_map_path, 'w'), 
                          indent=4)


    def Create_UnionClass_AtomClass_Rectiry_Map(self):
        # This functions relies on atom map json, so please run 'Create_AtomClassLabelMap' first.
        
        # This local function analyze the union class name and returns the major atom class name.
        def analyze_major_class(atom_class_list:Iterable, union_label_name:str) -> str:
            assert union_label_name.startswith('union_'), f'Only able to analyze union class name, but got {union_label_name}'
            # Search how many times for each atom_class appears in union_label_name
            sub_class_count = {sub_class:0 for sub_class in atom_class_list}
            for sub_class in atom_class_list:
                sub_class_count[sub_class] += union_label_name.count(sub_class)
            # Get the key order index with the max value in sub_class_count
            major_sub_class_name = max(sub_class_count, key=lambda k: sub_class_count[k])
            return major_sub_class_name

        def rectify_one_dataset_class_map(union_class_map:Dict, atom_class_map:Dict) -> Dict:
            # Create class rectify map for non-union labels
            class_names = union_class_map.keys()
            atom_class_names = [name for name in class_names \
                                if not name.startswith('union_')]
            atom_class_map = {
                class_name : new_atom_index \
                for new_atom_index, class_name in enumerate(atom_class_names)}
            class_rectify_map = {
                 union_class_map[atom_class] : atom_class_map[atom_class] \
                 for atom_class in atom_class_map.keys() \
                 if union_class_map[atom_class] != atom_class_map[atom_class]}
            
            # Create class rectify map for union labels
            for class_name in class_names:
                if class_name.startswith('union_'):
                    major_class_name = analyze_major_class(atom_class_map.keys(), class_name)
                    class_rectify_map[union_class_map[class_name]] = atom_class_map[major_class_name]
                else:
                    pass # The atom classes rectify map has already beed created above.

            return class_rectify_map

        # Main Creation Loop
        for modality in os.listdir(self.ModalityRootAfterInit):
            dataset_root = os.path.join(self.ModalityRootAfterInit, modality)
            if not os.path.isdir(dataset_root): continue
            datasets = os.listdir(dataset_root)
            for dataset in tqdm(datasets, desc=modality, total=len(datasets)):
                dataset_path = os.path.join(dataset_root, dataset)
                if not os.path.isdir(dataset_path): continue
                rectify_map_path = os.path.join(dataset_path, f'{dataset}_union_atom_class_rectify_map.json')
                
                union_class_map = json.load(open(os.path.join(
                    dataset_path, f'{dataset}_exist_class_map.json'), 'r'))
                atom_classes_map = json.load(open(os.path.join(
                    dataset_path, f'{dataset}_atom_class_map.json'), 'r'))
                class_rectify_map = rectify_one_dataset_class_map(union_class_map, atom_classes_map)
                json.dump(class_rectify_map,
                          open(rectify_map_path, 'w'),
                          indent=4)


    def Create_NumCase_AvgSlice_Map(self):
        Slices_Dict = json.loads(
            open(os.path.join(self.ModalityRootAfterInit, f'NumSlice_map.json'), 'r').read())
        modality_num_slice_map = {'_Num_Cases': {}, '_Avg_Slices_Per_Case': {}}
        
        for modality in tqdm(os.listdir(self.ModalityRootAfterInit), desc='Modality'):
            dataset_root = os.path.join(self.ModalityRootAfterInit, modality)
            if not os.path.isdir(dataset_root): continue
            
            dataset_num_cases_map = {'Num_Cases':{}, 'Avg_Slices_Per_Case': {}}
            for dataset in tqdm(os.listdir(dataset_root), desc=modality):
                case_root = os.path.join(dataset_root, dataset)
                if not os.path.isdir(case_root): continue
                count = 0
                
                for Case in os.listdir(case_root):
                    direction_root = os.path.join(case_root,Case)
                    if not os.path.isdir(direction_root): continue
                    count += len(os.listdir(direction_root))
                
                num_slices_dataset = Slices_Dict[modality][dataset]
                dataset_num_cases_map['Num_Cases'][dataset] = count
                dataset_num_cases_map['Avg_Slices_Per_Case'][dataset] = round(num_slices_dataset / count, 2)
                
            modality_num_slice_map['_Num_Cases'][modality] = sum(dataset_num_cases_map['Num_Cases'].values())
            modality_num_slice_map['_Avg_Slices_Per_Case'][modality] = np.mean(list(dataset_num_cases_map['Avg_Slices_Per_Case'].values()))
            modality_num_slice_map[modality] = dataset_num_cases_map
            json.dump(dataset_num_cases_map, 
                      open(os.path.join(dataset_root, f'{modality}_NumCase_AvgSlice.json'), 'w'), 
                      indent=4)
        json.dump(modality_num_slice_map, 
                  open(os.path.join(self.ModalityRootAfterInit, f'NumCase_AvgSlice.json'), 'w'), 
                  indent=4)


    def Calculate_Dataset_Value_Distributions(self):
        pool = futures.ProcessPoolExecutor(20)
        pbar = Progress(transient=True)
        pbar.start()
        
        modalities = os.listdir(self.ModalityRootAfterInit)
        pbar_modality = pbar.add_task('[red]Calculating', total=len(modalities))
        for modality in modalities:
            dataset_root = os.path.join(self.ModalityRootAfterInit, modality)
            datasets = os.listdir(dataset_root)
            pbar_dataset = pbar.add_task(f'[yellow]{modality}', total=len(datasets))
            if not os.path.isdir(dataset_root): continue
            
            for dataset in datasets:
                case_root = os.path.join(dataset_root, dataset)
                save_path = os.path.join(case_root, f'{dataset}_Distributions.json')
                # if os.path.exists(save_path): continue  # 跳过已经生成好的数据集
                cases = os.listdir(case_root)
                random.shuffle(cases)
                cases = cases[:100]
                pbar_case = pbar.add_task(f'[green]{dataset}', total=len(cases))
                if not os.path.isdir(case_root): continue

                case_list = []
                for Case in cases:
                    direction_root = os.path.join(case_root,Case)
                    if not os.path.isdir(direction_root): continue
                    
                    slice_list = []
                    for direction in os.listdir(direction_root):
                        slice_root = os.path.join(direction_root, direction)
                        if not os.path.isdir(slice_root): continue
                        
                        slices = os.listdir(slice_root)
                        random.shuffle(slices)
                        slices = slices[:50]
                        for slice in slices:
                            if slice.endswith('.png') and slice.startswith('image_'): 
                                distribution_one_slice = pool.submit(self.calculate_distributions_for_one_slice, 
                                                                     os.path.join(slice_root, slice))
                                slice_list.append(distribution_one_slice)
                    
                    # [slice_num, 4, 3]
                    slice_list = np.array([slice.result() for slice in slice_list])
                    # [4, 3]
                    case_result = np.array([slice_list[:, 0].min(axis=0), 
                                            slice_list[:, 1].max(axis=0), 
                                            slice_list[:, 2].mean(axis=0), 
                                            slice_list[:, 3].mean(axis=0)])
                    case_list.append(case_result)
                    pbar.advance(pbar_case)
                pbar.remove_task(pbar_case)
                
                # [case_num, 4, 3]
                case_list = np.array(case_list)
                dataset_wise_distributions = {
                    'min': case_list[:, 0].min(axis=0).tolist(),
                    'max': case_list[:, 1].max(axis=0).tolist(),
                    'mean': case_list[:, 2].mean(axis=0).tolist(),
                    'std': case_list[:, 3].mean(axis=0).tolist()
                }
                json.dump(dataset_wise_distributions, 
                          open(save_path, 'w'), 
                          indent=4)
                pbar.advance(pbar_dataset)
            pbar.remove_task(pbar_dataset)
            
            pbar.advance(pbar_modality)
        pbar.remove_task(pbar_modality)


    def Calculate_Dataset_Label_Distributions(self):
        def register_label(registry_dict, unique_labels:np.ndarray):
            for label in unique_labels:
                label = int(label)
                if label in registry_dict:
                    registry_dict[label] += 1
                else:
                    registry_dict[label] =  1
            return registry_dict
        
        pool = futures.ProcessPoolExecutor(20)
        pbar = Progress(transient=True)
        pbar.start()
        

        modalities = os.listdir(self.ModalityRootAfterInit)
        pbar_modality = pbar.add_task('[red]Calculating', total=len(modalities))
        for modality in modalities:
            dataset_root = os.path.join(self.ModalityRootAfterInit, modality)
            if not os.path.isdir(dataset_root): 
                pbar.advance(pbar_modality)
                continue
            datasets = os.listdir(dataset_root)
            pbar_dataset = pbar.add_task(f'[yellow]{modality}', total=len(datasets))

            for dataset in datasets:
                dataset_label_distributions = {}
                case_root = os.path.join(dataset_root, dataset)
                save_path = os.path.join(case_root, f'{dataset}_Label_Distributions.json')
                # if os.path.exists(save_path): continue  # ���过已经生成好的数据集
                if not os.path.isdir(case_root) or os.path.exists(save_path): 
                    pbar.advance(pbar_dataset)
                    continue
                cases = os.listdir(case_root)
                # random.shuffle(cases)
                # cases = cases[:500]
                pbar_case = pbar.add_task(f'[green]{dataset}', total=len(cases))
                
                for Case in cases:
                    direction_root = os.path.join(case_root, Case)
                    if not os.path.isdir(direction_root): continue

                    slice_list = []
                    for direction in os.listdir(direction_root):
                        slice_root = os.path.join(direction_root, direction)
                        if not os.path.isdir(slice_root): continue

                        slices = os.listdir(slice_root)
                        random.shuffle(slices)
                        slices = slices[:100]
                        for slice in slices:
                            if slice.endswith('.png') and slice.startswith('mask_'):
                                label_distribution_one_slice = pool.submit(
                                    self.calculate_label_distributions_for_one_slice,
                                    os.path.join(slice_root, slice))
                                slice_list.append(label_distribution_one_slice)

                    for slice in slice_list:
                        register_label(dataset_label_distributions, slice.result())
                    pbar.advance(pbar_case)
                
                sorted_dict = dict(sorted(dataset_label_distributions.items(), key=lambda x: x[0]))
                json.dump(sorted_dict, open(save_path, 'w'), indent=4)
                
                pbar.remove_task(pbar_case)
                pbar.advance(pbar_dataset)
            
            pbar.remove_task(pbar_dataset)
            pbar.advance(pbar_modality)
        
        pbar.remove_task(pbar_modality)


    def Hierarchical_Folder_init(self):
        # self.init_image()
        self.init_mask()


    def init_image(self):
        print('正在获取文件列表')
        image_list = os.listdir(self.image_path)
        num_image = len(image_list)
        print(f'文件列表已获取, images: {num_image}')
        
        # 图像合并
        try:
            for index in tqdm(range(num_image), desc=f'Processing Images', dynamic_ncols=True, miniters=1000, mininterval=2):
                file_meta_now = self.analyze_file_name(image_list[index])
                # 确定目标路径
                save_root = os.path.join(self.ModalityRootAfterInit, 
                                         file_meta_now['modality_sub-modality'],
                                         file_meta_now['dataset name'],
                                         file_meta_now['ori name'],
                                         file_meta_now['slice_direction'])
                if not os.path.exists(save_root): os.makedirs(save_root)
                # 执行移动
                # tqdm.write(f'{image_list[index]} --> {save_root}')
                shutil.move(os.path.join(self.image_path, image_list[index]), 
                            os.path.join(save_root, f"image_{file_meta_now['slice_index']}.png"))
                
        except KeyboardInterrupt:
            pdb.set_trace()


    def init_mask(self):
        def is_same(file_meta_1, file_meta_2, level):
            if file_meta_1 == {} or file_meta_2 == {}: return False
            # 前三项相同即为同一个序列
            for i in self.MASK_NAME_DEFINITION[:level]:
                if file_meta_1[i] != file_meta_2[i]: return False
            return True

        print('正在获取文件列表')
        mask_list = os.listdir(self.mask_path)
        num_mask_files = len(mask_list)
        print(f'文件列表已获取, masks: {num_mask_files}')

        pool_process_1 = futures.ProcessPoolExecutor(max_workers=20)
        pool_process_2 = futures.ProcessPoolExecutor(max_workers=20)
        cache_slice_segmentation = []	# (class_id, ndarray)  用于存储同一横断面上的所有分割实例
        cache_save_result:List[Future[bool]] = []
        file_meta_last:Dict[str, str] = {}

        try:
            for index in tqdm(range(num_mask_files), desc='Processing Masks', 
                                dynamic_ncols=True, miniters=1000, mininterval=2):
                file_meta_now = self.analyze_file_name(mask_list[index])
                
                # 定期清空多进程返回池，防止返回结果溢出，也能保证多线程队列不溢出。
                if len(cache_save_result) // 1000 > 0:
                    for result in cache_save_result: result.result()
                    cache_save_result:List[Future[bool]] = []
                
                # 如果发现了新的slice, 生成一个slice的mask
                if (not is_same(file_meta_last, file_meta_now, -2) and len(cache_slice_segmentation)>0) or index==num_mask_files-1:
                    mask_of_slice = self.instance_to_segmentation(cache_slice_segmentation)
                    cache_slice_segmentation = []
                    sava_path = os.path.join(
                        self.ModalityRootAfterInit,
                        file_meta_last['modality_sub-modality'],
                        file_meta_last['dataset name'],
                        file_meta_last['ori name'],
                        file_meta_last['slice_direction'],
                        f"mask_{file_meta_last['slice_index']}.png"
                    )
                    # 以png格式存储
                    save_result = pool_process_2.submit(cv2.imwrite, sava_path, mask_of_slice, [cv2.IMWRITE_PNG_COMPRESSION,5])
                    cache_save_result.append(save_result)

                file_meta_last = file_meta_now
                slice_segmentation_result = pool_process_1.submit(self.png_to_ndarray, os.path.join(self.mask_path, mask_list[index]))
                cache_slice_segmentation.append((file_meta_now['class_instance'], slice_segmentation_result))
        
        except KeyboardInterrupt:
            pdb.set_trace()



if __name__ == '__main__':
    ROOT_SA_Med2D_16M = 'D:/PostGraduate/DL/SA-Med2D-20M/'
    DEST_ROOT = 'D:/PostGraduate/DL/SA-Med2D-20M'
    processor = DataStructurer(ROOT_SA_Med2D_16M, DEST_ROOT, 'init')
    processor.Create_NumCase_AvgSlice_Map()

