import os
import pdb
import json

import numpy as np
import SimpleITK as sitk

from mgamdata.process.PreCrop_3D import PreCropper3D, RandomCrop3D
from mgamdata.dataset.BraTs2024.meta import BraTs2024_MODALITIES


class BraTs2024Cropper(PreCropper3D):
    def parse_task(self) -> list[tuple[RandomCrop3D, str, str, int, str]]:
        """
        Task List, each task contains:
            - RandomCrop3D Class
            - case_dir
            - save_folder
        """
        task_list = []
        source_mha_folder = self.args.source_mha_folder
        dest_npz_folder = self.args.dest_npz_folder

        for phase in ["train", "val"]:
            phase_dir = os.path.join(source_mha_folder, phase)
            series_folders = [os.path.join(phase_dir, i) 
                              for i in os.listdir(phase_dir)
                              if os.path.isdir(os.path.join(phase_dir, i))]

            for series_folder in series_folders:
                series_name = os.path.basename(series_folder)
                save_folder = os.path.join(dest_npz_folder, series_name)
                cropper = RandomCrop3D(
                    self.args.crop_size,
                    cat_max_ratio=self.args.crop_cat_max,
                    ignore_index=self.args.ignore_index,
                )
                task_list.append((
                    cropper, 
                    series_folder, 
                    save_folder))
        return task_list

    def _cut_edge(self, image_array, anno_array):
        # Cut Edge on each dimension if it contains Non-Zero value
        if self.args.cut_edge is not None and any(self.args.cut_edge):
            for dim, cut_length in enumerate(self.args.cut_edge):
                dim_length = image_array.shape[dim]
                indices_to_cut = np.concatenate(
                    [
                        np.arange(cut_length),
                        np.arange(dim_length - cut_length, dim_length),
                    ]
                )
                image_array = np.delete(image_array, indices_to_cut, axis=dim)
                if anno_array is not None:
                    anno_array = np.delete(anno_array, indices_to_cut, axis=dim)
        return image_array, anno_array

    def crop_per_series(self, args: tuple) -> dict:
        cropper, series_folder, save_folder = args
        cropper: RandomCrop3D
        os.makedirs(save_folder, exist_ok=True)
        existed_classes = {}
        cropped_center = []
        
        # get an example image
        modality_images = {
            modality: os.path.join(series_folder, f"{modality}.mha")
            for modality in BraTs2024_MODALITIES
            if os.path.exists(os.path.join(series_folder, f"{modality}.mha"))
        }
        label_path = os.path.join(series_folder, "label.mha")
        if not os.path.exists(label_path):
            label_path = None
            
        first_image_path = next(iter(modality_images.values()))

        for crop_idx, (img_array, anno_array, crop_bbox) in enumerate(
            self.Crop3D(cropper, first_image_path, label_path)
        ):
            cropped_images = {}
            for modality, image_path in modality_images.items():
                image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image)
                image = self._cut_edge(image, None)[0]
                cropped_images[modality] = cropper.crop(image, crop_bbox)
            
            save_path = os.path.join(save_folder, f"{crop_idx}.npz")
            np.savez_compressed(
                file=save_path,
                **cropped_images,
                gt_seg_map=anno_array if anno_array is not None else np.nan,
            )

            existed_classes[os.path.basename(save_path)] = (
                np.unique(anno_array).tolist() 
                if anno_array is not None else None
            )
            z1,z2,y1,y2,x1,x2 = crop_bbox
            cropped_center.append(((z1+z2)/2, (y1+y2)/2, (x1+x2)/2))

        num_patches = len(existed_classes)
        anno_available = label_path is not None and num_patches > 0

        self._draw_cropped_center(cropped_center, save_folder)
        json.dump(
            {
                "series_id": os.path.basename(save_folder),
                "shape": img_array.shape if num_patches > 0 else None,
                "num_patches": num_patches,
                "anno_available": anno_available,
                "class_within_patch": existed_classes,
                "cropped_center": cropped_center,
            },
            open(
                os.path.join(save_folder, "SeriesMeta.json"),
                "w",
            ),
            indent=4,
        )
        path_index = [
            os.path.relpath(
                os.path.join(save_folder, sample_basepath), os.path.dirname(save_folder)
            )
            for sample_basepath in existed_classes.keys()
        ]
        
        return {
            os.path.basename(save_folder): {
                "num_patches": num_patches,
                "anno_available": anno_available,
                "sample_paths": path_index,
                "cropped_center": cropped_center,
            }
        }


if __name__ == '__main__':
    BraTs2024Cropper()