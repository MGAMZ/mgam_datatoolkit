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
        from mgamdata.process.GeneralPreProcess import RandomCrop3D

        task_list = []
        for series in os.listdir(self.args.source_mha_folder):
            image_mha_path = os.path.join(self.args.source_mha_folder, series, "ct.mha")
            label_mha_path = os.path.join(self.args.source_mha_folder, series, "segmentations.mha")
            task_list.append(
                (
                    RandomCrop3D(
                        self.args.crop_size,
                        self.args.crop_cat_max,
                        self.args.ignore_index,
                    ),
                    image_mha_path,
                    label_mha_path if os.path.exists(label_mha_path) else None,
                    os.path.join(self.args.dest_npz_folder, series),
                )
            )
        return task_list


if __name__ == "__main__":
    Totalsegmentator_PreCrop()
