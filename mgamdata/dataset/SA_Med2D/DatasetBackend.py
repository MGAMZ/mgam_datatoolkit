import logging
import os
import pdb
import random
import colorama
colorama.init()
from warnings import warn
from abc import abstractmethod
from typing import Dict, List, Tuple, Iterable, Sequence

import orjson
import tqdm
from mmengine.logging import print_log
from mmengine.utils import ManagerMixin
from mmseg.datasets import BaseSegDataset

from .DatasetInitialize import SA_Med2D
import numpy as np


DATASET_ROOT_LINUX = '/file1/mgam_datasets/SA-Med2D-20M/CaseSeperated/'
DATASET_ROOT_MNT = '/file1/mgam_datasets/SA-Med2D-20M/CaseSeperated/'


# 单进程 mmseg数据集设定
class SA_Med2D_Dataset(SA_Med2D, BaseSegDataset):
    DATASET_SPLIT_RATIO = [0.7, 0.75, 1.0]
    
    def __init__(self,
                 modality:str,
                 dataset_source:str,
                 split:str,
                 debug:bool=False,
                 activate_case_ratio:float|None=None,
                 union_atom_rectify:bool=False,
                 root_path_mode:str='from_linux',
                 deterministic:bool=False, # 可复现的采样
                 **kwargs,
        ):
        assert split in ['train', 'val', 'test']
        self.split = split
        assert modality in self.MODALITIES
        self.modality = modality
        self.dataset_source = dataset_source
        self.debug = debug
        self.deterministic = deterministic
        self.activate_case_ratio = activate_case_ratio
        self.structured_npz_root = self._structured_npz_root(root_path_mode)
        self.selected_dataset_root = os.path.join(self.structured_npz_root, modality, dataset_source)
        assert os.path.exists(self.selected_dataset_root),\
            f'Selected {modality} and {dataset_source} does not exist: {self.selected_dataset_root}'
        # 加载索引
        self.case_slice_map, self.dataset_distributions = self._init_index_map(
            self.selected_dataset_root, dataset_source)
        # 全局共享数据集参数
        union_atom_map, label_map, proxy, self.union_atom_map_path= self.set_proxy(
            modality, dataset_source, union_atom_rectify)
        # mmseg标准数据集接口配置
        super(SA_Med2D_Dataset, self).__init__(
            metainfo={'classes':list(label_map.keys())}, 
            img_suffix='.png', seg_map_suffix='.png', **kwargs)

    @staticmethod
    def _structured_npz_root(root_path_mode:str):
        if root_path_mode=='from_linux':
            structured_npz_root = DATASET_ROOT_LINUX
        elif root_path_mode=='from_mnt':
            structured_npz_root = DATASET_ROOT_MNT
        else:
            raise NotImplementedError(f"Unknown root_path_mode: {root_path_mode}, should be either 'from_linux' or 'from_mnt'.")
        return structured_npz_root

    @staticmethod
    def _init_index_map(selected_dataset_root, dataset_source):
        # 逐case的slice数量映射表
        case_slice_map:Dict[str, Dict[str, List[str]]] = orjson.loads(
            open(os.path.join(selected_dataset_root, 
                              f'{dataset_source}_CaseSlice_map.json'
                ), 'r').read())
        # 加载数据集分布统计量
        dataset_distributions:Dict[str, float] = orjson.loads(
            open(os.path.join(selected_dataset_root, 
                              f'{dataset_source}_Distributions.json'
                ), 'r').read())
        return case_slice_map, dataset_distributions

    @classmethod
    def set_proxy(cls, modality, dataset_source, union_atom_rectify):
        if os.path.exists(os.path.join(cls._structured_npz_root('from_mnt'), modality, dataset_source)):
            selected_dataset_root = os.path.join(cls._structured_npz_root('from_mnt'), modality, dataset_source)
        elif os.path.exists(os.path.join(cls._structured_npz_root('from_linux'), modality, dataset_source)):
            selected_dataset_root = os.path.join(cls._structured_npz_root('from_linux'), modality, dataset_source)
        else:
            raise FileNotFoundError(f"DataSource Not Available. Modality:{modality} | dataset name: {dataset_source} | union_atom_rectify:{union_atom_rectify}")
        
        if union_atom_rectify: # 可选纠正SA-Med2D数据集中的大量异常Union Label
            label_map = orjson.loads(
                open(os.path.join(selected_dataset_root, 
                f"{dataset_source}_atom_class_map.json"), 'r').read())
            union_atom_map_path = os.path.join(
                selected_dataset_root, 
                f'{dataset_source}_union_atom_class_rectify_map.json')
            union_atom_map:Dict[str,int] = orjson.loads(
                open(union_atom_map_path, 'r').read())
            # Perform rectify
            for old_label_name, old_label_idx in label_map.items():
                if union_atom_map.get(str(old_label_idx), None) is not None:
                    label_map[old_label_name] = union_atom_map[str(old_label_idx)]
        else:
            union_atom_map = None
            label_map = orjson.loads(open(
                os.path.join(selected_dataset_root, 
                f"{dataset_source}_exist_class_map.json"), 'r').read())
        if DatasetBackend_GlobalProxy.check_instance_created('union_atom_map'):
            proxy = DatasetBackend_GlobalProxy.get_instance('union_atom_map')
            proxy.union_atom_map = union_atom_map
            proxy.atom_classes = label_map
        else:
            proxy = DatasetBackend_GlobalProxy.get_instance(
                name='union_atom_map', 
                union_atom_map=union_atom_map,
                atom_classes=label_map)
        return union_atom_map, label_map, proxy, union_atom_map_path

    @classmethod
    def split_dataset(cls, map_dict:Dict, split:str) -> List:
        cases_keys = list(map_dict.keys())
        num_cases = len(cases_keys)
        range = [0, 
                 int(num_cases * cls.DATASET_SPLIT_RATIO[0]), 
                 int(num_cases * cls.DATASET_SPLIT_RATIO[1]), 
                 int(num_cases * cls.DATASET_SPLIT_RATIO[2])]
        if split == 'train':
            return cases_keys[range[0]:range[1]-1]
        elif split == 'val':
            return cases_keys[range[1]-1:range[2]]
        elif split == 'test':
            return cases_keys[range[2]:range[3]]
        else:
            raise ValueError(f'{split} is not supported.')


    def sampling_from_set(self, sample_set:Sequence, num_samples:int) -> Iterable:
        if self.deterministic:
            return sample_set[:num_samples]
        else:
            if isinstance(sample_set, np.ndarray):
                return np.random.choice(sample_set, size=num_samples, replace=False)
            else:
                return random.sample(sample_set, num_samples)


    def key_to_sample_path(self, key:str) -> Tuple[str, str]:
        key_without_image_mask_prefix = key.split('/')[-1]
        file_name_components = self.analyze_file_name(key_without_image_mask_prefix)
        case_root = os.path.join(self.structured_npz_root, 
                                 file_name_components['modality_sub-modality'],
                                 file_name_components['dataset name'],
                                 file_name_components['ori name'])
        image_name = file_name_components['slice_direction'] + '.npz'
        mask_name = file_name_components['slice_direction'] + '_mask.npz'
        return (os.path.join(case_root, image_name), os.path.join(case_root, mask_name))


    def slice_series_fetcher(self):
        # 在Case维度，三切分数据集
        used_case_names = self.split_dataset(self.case_slice_map, self.split)
        # 可选 Cases-Wise Filtering
        if (self.split != 'test') and (self.activate_case_ratio is not None) and (self.activate_case_ratio<1.0): 
            assert 0<self.activate_case_ratio<=1, f'activate_case_ratio must be in (0, 1].'
            target_num_used_cases = int(len(used_case_names) * self.activate_case_ratio)
            if target_num_used_cases == 0:
                target_num_used_cases = 1
            used_case_names = self.sampling_from_set(used_case_names, target_num_used_cases)
        
        for Case in tqdm.tqdm(used_case_names, desc=f"SA-Med2D | {self.modality} | {self.dataset_source} | {self.split}"):
            direction_root = os.path.join(self.selected_dataset_root, Case)
            if not os.path.isdir(direction_root): continue
            
            for direction in os.listdir(direction_root):
                slice_root = os.path.join(direction_root, direction)
                if not os.path.isdir(slice_root): continue
                if not os.path.exists(slice_root): raise FileNotFoundError(f'{slice_root} does not exist.')

                avail_idx =  self.case_slice_map[Case][direction]
                yield (slice_root, Case, direction, avail_idx)

    @abstractmethod
    def load_data_list(self) -> List[Dict]:
        raise NotImplementedError



class SA_Med2D_Dataset_NormalSample(SA_Med2D_Dataset):
    def __init__(self, num_slices_per_sample, *args, **kwargs):
        self.num_slices_per_sample = num_slices_per_sample
        super().__init__(*args, **kwargs)
    
    def _construct_sample_dict(self, slice_root, idx):
        data_info = {
            'img_path':           (slice_root, idx),  # 中心image
            'seg_map_path':       (slice_root, idx),  # 中心label
            'seg_fields':         [],
            'reduce_zero_label':  self.reduce_zero_label,
            }
        data_info.update(self.dataset_distributions)
        if hasattr(self, 'union_atom_map_path'):
            data_info['union_atom_map_path'] = self.union_atom_map_path
        return data_info
    
    def load_data_list(self) -> List[Dict]:
        samples = []
        for (slice_root, Case, direction, avail_idx) in self.slice_series_fetcher():
            used_idx = self.sampling_from_set(avail_idx, self.num_slices_per_sample)
            for idx in used_idx:
                samples.append(self._construct_sample_dict(slice_root, idx))

        return samples



class SA_Med2D_Dataset_MultiSliceSample(SA_Med2D_Dataset):
    def __init__(self, 
                 num_images_per_sample:int,
                 num_labels_per_sample:int,
                 stride:int|None,    # 滑动窗口采样
                 slice_gap:int,
                 max_sample_per_case:int|None=None, # 逐case恒定样本数采样
                 *args, **kwargs):
        # 参数检查
        if stride is not None:
            assert stride > 0 and isinstance(stride, int),\
                (f"If shift window sampling is deployed,"
                 f"stride must be positive and int, but got {stride}")
            assert max_sample_per_case is None,\
                "Either shift window sampling or fixed per-case sampling is available."
        else:
            if isinstance(max_sample_per_case, float):
                assert 0<max_sample_per_case<=1,\
                    f"If max_sample_per_case is float, it must be in (0, 1]. Now got {max_sample_per_case}"
            elif isinstance(max_sample_per_case, int):
                assert 0<max_sample_per_case>=1,\
                    f"If max_sample_per_case is int, it must be >= 1. Now got {max_sample_per_case}"
            else:
                raise ValueError(f"If fixed per-case sampling is deployed,"
                                 f"stride must be positive and int, "
                                 f"but got {max_sample_per_case}")
        assert slice_gap >= 1 and isinstance(slice_gap, int),\
            f'slice_gap must be positive and int, but got {slice_gap}'
        assert num_images_per_sample%2==1 and num_labels_per_sample%2==1,\
            f"only singular num of slice is available."
        
        # 由于label在mmseg的数据流中是(H,W)的定义，且有专用的封装格式，同时处理batch和num_slice_per_sample会给Preprocessor的实现带来困难，较为复杂
        # 故多个Label输入的情况，会产生警告。
        if num_labels_per_sample > 1:
            message = f'You are loading {num_labels_per_sample} label for each sample, which is NOT the standard work assumption.'
            warn(colorama.Fore.YELLOW + message + colorama.Style.RESET_ALL)
        self.num_images_per_sample = num_images_per_sample
        self.num_labels_per_sample = num_labels_per_sample
        self.stride = stride
        self.slice_gap = slice_gap
        self.max_sample_per_case = max_sample_per_case
        super().__init__(*args, **kwargs)
    
    
    def _construct_sample_dict(self, slice_root, avail_idx, center_idx, image_idxs, label_idxs):
        data_info = {
            'img_path':           (slice_root, avail_idx[center_idx]),  # 中心image
            'seg_map_path':       (slice_root, avail_idx[center_idx]),  # 中心label
            'multi_img_path':     image_idxs,
            'multi_seg_map_path': label_idxs,
            'seg_fields':         [],
            'reduce_zero_label':  self.reduce_zero_label,
            }
        data_info.update(self.dataset_distributions)
        if hasattr(self, 'union_atom_map_path'):
            data_info['union_atom_map_path'] = self.union_atom_map_path
        
        return data_info
    
    
    def load_data_list(self) -> List:
        # 最大采样范围受限于要求的image和label中数量更大的那个
        # 采样时，两个集合中心对齐，两侧对称
        max_gap_to_center = (max(self.num_images_per_sample, self.num_labels_per_sample) // 2) * self.slice_gap
        mmseg_sample_list = []
        
        # 获取一个扫描序列
        for (slice_root, Case, direction, avail_idx) in self.slice_series_fetcher():
            num_samples = len(avail_idx)
            
            # 确定可用的中心slice序号
            if self.max_sample_per_case is not None:
                avail_center_idx = range(
                    max_gap_to_center, 
                    num_samples - max_gap_to_center)
                if len(avail_center_idx) == 0:
                    continue
                
                if self.split != 'test':
                    if isinstance(self.max_sample_per_case, int):
                        num_samples = min(self.max_sample_per_case, len(avail_center_idx))
                    elif isinstance(self.max_sample_per_case, float):
                        num_samples = int(len(avail_center_idx) * self.max_sample_per_case)
                    else:
                        raise TypeError(f'Unexpected type of max_sample_per_case: {self.max_sample_per_case}')
                    try:
                        avail_center_idx = self.sampling_from_set(
                            avail_center_idx, max(num_samples, 1))
                    except:
                        raise RuntimeError(
                            f"Unable to Sample. "
                            f"series_root:{slice_root} | "
                            f"series_case:{Case} | "
                            f"series_dire:{direction} | "
                            f"series_avail_idx:{avail_idx} | "
                            f"need_num_samples:{num_samples}")
            
            else:
                assert self.stride is not None, "Either shift window sampling or fixed per-case sampling is available."
                avail_center_idx = range(max_gap_to_center,
                    num_samples - max_gap_to_center - 1, # range的stop位置不取
                    self.stride if self.split != 'test' else 1)
                if len(avail_center_idx) == 0:
                    continue
            
            # 对每个可用的中心点，在其两侧对称采样
            for center_idx in avail_center_idx:
                negative_image_idx  = center_idx - self.num_images_per_sample//2 * self.slice_gap
                positive_image_idx = center_idx + self.num_images_per_sample//2 * self.slice_gap
                image_idx_of_this_sample = avail_idx[negative_image_idx : positive_image_idx+1 : self.slice_gap]
                negative_label_idx  = center_idx - self.num_labels_per_sample//2 * self.slice_gap
                positive_label_idx = center_idx + self.num_labels_per_sample//2 * self.slice_gap
                label_idx_of_this_sample = avail_idx[negative_label_idx : positive_label_idx+1 : self.slice_gap]
                
                sample = self._construct_sample_dict(
                    slice_root, avail_idx, center_idx, 
                    image_idx_of_this_sample, label_idx_of_this_sample)
                mmseg_sample_list.append(sample)
        
        if len(mmseg_sample_list) == 0:
            raise FileNotFoundError(
                f"No samples are available for "
                f"SA-Med2D | {self.modality} | {self.dataset_source} | {self.split}")
        
        print_log(msg=f"SA-Med2D | {self.modality} | {self.dataset_source} | {self.split} | Num Samples: {len(mmseg_sample_list)}",
                  logger='current', level=logging.INFO)
        class_map:Dict = self._metainfo['classes']
        print_log(msg=f"SA-Med2D | {self.modality} | {self.dataset_source} | {len(class_map)} atom classes are {class_map}",
                  logger='current', level=logging.INFO)
        
        if self.debug:
            if self.split == 'train':
                return mmseg_sample_list[:32]
            else:
                return mmseg_sample_list[:2]
        else:
            return mmseg_sample_list



class DatasetBackend_GlobalProxy(ManagerMixin):
    def __init__(self, name:str, union_atom_map=None, 
                 atom_classes=None, union_classes=None) -> None:
        super().__init__(name)
        self.union_atom_map = union_atom_map
        self.atom_classes = atom_classes
        self.union_classes = union_classes





