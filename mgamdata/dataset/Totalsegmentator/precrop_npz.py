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

from mgamdata.process.PreCrop_3D import PreCropper3D



class Totalsegmentator_PreCrop(PreCropper3D):
    def parse_task(self):
        from mgamdata.mm.mmseg_Dev3D import RandomCrop3D
        task_list = []
        for series in os.listdir(self.args.source_mha_folder):
            task_list.append((
                RandomCrop3D(self.args.crop_size, self.args.crop_cat_max, self.args.ignore_index),
                os.path.join(self.args.source_mha_folder, series, 'ct.mha'),
                os.path.join(self.args.source_mha_folder, series, 'segmentations.mha'),
                os.path.join(self.args.dest_npz_folder, series)))
        return task_list



if __name__ == '__main__':
    Totalsegmentator_PreCrop()
