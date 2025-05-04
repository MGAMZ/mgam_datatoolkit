import os
import argparse
import json
import pdb
import multiprocessing as mp
from abc import abstractmethod
from colorama import Fore, Style
from typing_extensions import deprecated
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..process.GeneralPreProcess import RandomCrop3D
from .NDArray import unsafe_astype


class PreCropper3D:
    def __init__(self):
        self.main()

    def arg_parse(self) -> argparse.ArgumentParser:
        argparser = argparse.ArgumentParser("Pre-Random-Crop 3D")
        argparser.add_argument("source_mha_folder", type=str, help="The folder containing mha files.")
        argparser.add_argument("dest_npz_folder", type=str, help="The folder to save npz files.")
        argparser.add_argument("--crop-size", type=int, nargs=3, required=True, 
                               help="The size of cropped volume.")
        argparser.add_argument("--crop-cat-max", type=float, default=1.0,
                               help="Max ratio for single catagory can occupy.")
        argparser.add_argument("--num-cropped-ratio", type=int, default=4,
                               help="The number of cropped volumes per series.")
        argparser.add_argument( "--ensure-index", type=int, default=None, nargs="+", 
                               help="The index to ensure in segmentation.")
        argparser.add_argument("--ensure-ratio", type=float, default=None,
                               help="The chance for an ensurance to perform.")
        argparser.add_argument("--ignore-index", type=int, default=255,
                               help="The index to ignore in segmentation. "
                               "It will not taken into consideration during "
                               "the determination of whether the cropped patch "
                               "meets the `crop-cat-max` setting.")
        argparser.add_argument("--mp", action="store_true", default=False, help="Whether to use multiprocessing.",)
        argparser.add_argument("--cut-edge", type=int, default=[1, 1, 1], nargs="+",
            help="The edge size to cut off (delete) in each dimension. "
            "This is to avoid some datasets have invalid slice on the edge of a dcm series.",
        )
        argparser.add_argument("--std-thr", type=float, default=None, help="The threshold for std to determine whether a slice is valid.",)
        return argparser

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

    def all_index_ensured(self, label: np.ndarray):
        if self.args.ensure_index is None or np.random.rand() > self.args.ensure_ratio:
            return True
        else:
            return any(index not in label for index in self.args.ensure_index)

    @staticmethod
    def _draw_cropped_center(cropped_center, save_folder):
        # 创建画布和网格布局
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2)
        
        # 左侧3D散点图 - 占据整个左列
        ax1 = fig.add_subplot(gs[:, 0], projection='3d')
        z_coords = [center[0] for center in cropped_center]
        y_coords = [center[1] for center in cropped_center]
        x_coords = [center[2] for center in cropped_center]
        
        ax1.scatter(x_coords, y_coords, z_coords, c='b', marker='o')
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.set_zlabel('Z axis')
        ax1.set_title('3D Distribution')
        
        # 右上 - XY投影
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(x_coords, y_coords, c='r', marker='o')
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_title('XY Projection')
        ax2.grid(True)
        
        # 右中 - YZ投影
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(y_coords, z_coords, c='g', marker='o')
        ax3.set_xlabel('Y axis')
        ax3.set_ylabel('Z axis')
        ax3.set_title('YZ Projection')
        ax3.grid(True)
        
        # 右下 - XZ投影
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.scatter(x_coords, z_coords, c='purple', marker='o')
        ax4.set_xlabel('X axis')
        ax4.set_ylabel('Z axis')
        ax4.set_title('XZ Projection')
        ax4.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(
            os.path.join(save_folder, "CroppedCenter.png"), 
            dpi=300, 
            bbox_inches='tight')
        plt.close()

    def crop_per_series(self, args: tuple) -> dict:
        cropper, image_itk_path, anno_itk_path, save_folder = args
        cropper: RandomCrop3D
        os.makedirs(save_folder, exist_ok=True)
        existed_classes = {}
        cropped_center = []

        for crop_idx, (img_array, anno_array, crop_bbox) in enumerate(
            self.Crop3D(cropper, image_itk_path, anno_itk_path)
        ):
            save_path = os.path.join(save_folder, f"{crop_idx}.npz")
            np.savez_compressed(
                file=save_path,
                img=img_array,
                gt_seg_map=anno_array if anno_array is not None else np.nan,
            )

            existed_classes[os.path.basename(save_path)] = (
                np.unique(anno_array).tolist() 
                if anno_array is not None else None
            )
            z1,z2,y1,y2,x1,x2 = crop_bbox
            cropped_center.append(((z1+z2)/2, (y1+y2)/2, (x1+x2)/2))

        num_patches = len(existed_classes)
        anno_available = anno_itk_path is not None and num_patches > 0

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

    def Crop3D(
        self, 
        cropper: RandomCrop3D, 
        image_itk_path: str, 
        anno_itk_path: str | None
    ):
        # Load Image and Segmentations
        image_itk_image = sitk.ReadImage(image_itk_path)
        image_itk_image = sitk.DICOMOrient(image_itk_image, 'LPI')
        image_array = sitk.GetArrayFromImage(image_itk_image)
        if anno_itk_path is not None:
            anno_itk_image = sitk.ReadImage(anno_itk_path)
            anno_itk_image = sitk.DICOMOrient(anno_itk_image, 'LPI')
            anno_array = sitk.GetArrayFromImage(anno_itk_image)
            assert (
                image_array.shape == anno_array.shape
            ), f"Shape Mismatch on {image_itk_image}, image shape: {image_array.shape}, anno shape: {anno_array.shape}"
        else:
            anno_array = None

        # Deprecate too small volume
        minimum_required_size = np.array(cropper.crop_size)
        if self.args.cut_edge is not None:
            minimum_required_size += np.array(self.args.cut_edge) * 2
        if np.any(np.array(image_array.shape) < minimum_required_size):
            tqdm.write(
                Fore.YELLOW + \
                f"Deprecated {image_itk_path} due to too small volume: "
                f"Minimum {minimum_required_size}, got {image_array.shape}" + \
                Style.RESET_ALL,
            )
            return None

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

        # not giving the seg_fields, RandomCrop3D will not crop label.
        data = {
            "img": image_array,
            "gt_seg_map": anno_array,
            "seg_fields": ["gt_seg_map"] if anno_array is not None else [],
        }
        
        # calculate the number of cropped patches
        num_cropped = (
            int(np.prod(np.array(image_array.shape) / np.array(cropper.crop_size)))
            * self.args.num_cropped_ratio
        )

        # start cropping
        for i in range(num_cropped):
            # if no label, can't check cat_max_ratio
            if anno_itk_path is None:
                cropper.cat_max_ratio = 1.0

            crop_bbox = cropper.crop_bbox(data)
            if crop_bbox is None:
                return
            
            if anno_itk_path is not None:
                cropped_ann: np.ndarray = cropper.crop(anno_array, crop_bbox)
                cropped_ann = unsafe_astype(cropped_ann, np.uint8)

                if self.all_index_ensured(cropped_ann):
                    cropped_img: np.ndarray = cropper.crop(image_array, crop_bbox)
                    cropped_img = unsafe_astype(cropped_img, np.int16)
                    yield cropped_img, cropped_ann, crop_bbox
                else:
                    tqdm.write(
                        f"deprecated due to failing to ensure index: {anno_itk_path} | crop_idx: {i}"
                    )

            else:
                cropped_img: np.ndarray = cropper.crop(image_array, crop_bbox)
                cropped_img = unsafe_astype(cropped_img, np.int16)
                yield cropped_img, None, crop_bbox

    def main(self):
        self.args = self.arg_parse().parse_args()
        
        os.makedirs(self.args.dest_npz_folder, exist_ok=True)
        crop_meta_path = os.path.join(self.args.dest_npz_folder, "crop_meta.json")
        json.dump(vars(self.args), open(crop_meta_path, "w"), indent=4)

        self.task_list = self.parse_task()
        series_meta = {}
        if self.args.mp:
            with mp.Pool() as pool:
                fetcher = pool.imap_unordered(self.crop_per_series, self.task_list)
                for result in tqdm(
                    fetcher,
                    "Cropping",
                    total=len(self.task_list),
                    dynamic_ncols=True,
                    leave=False,
                ):
                    series_meta.update(result)
        else:
            for task in tqdm(
                self.task_list, "Cropping", dynamic_ncols=True, leave=False
            ):
                result = self.crop_per_series(task)
                series_meta.update(result)

        print(f"Finished cropping {len(self.task_list)} series.")
        print(f"Writing cropped patches' meta to {crop_meta_path}.")

        cropped_series_meta = {
            "crop_args": vars(self.args),
            "num_series": len(series_meta),
            "num_patches": sum(
                [
                    one_series_meta["num_patches"]
                    for one_series_meta in series_meta.values()
                ]
            ),
            "anno_available": [
                series_id
                for series_id, series_meta in series_meta.items()
                if series_meta["anno_available"] is True
            ],
        }
        json.dump(cropped_series_meta, open(crop_meta_path, "w"), indent=4)


@deprecated("Use the flexible version `SemiSupervisedMhaCropper3D` please.")
class StandardMhaCropper3D(PreCropper3D):
    def parse_task(self):
        task_list = []
        image_mha_folder = os.path.join(self.args.source_mha_folder, "image")
        label_mha_folder = os.path.join(self.args.source_mha_folder, "label")

        for series in os.listdir(image_mha_folder):
            if series.endswith(".mha"):
                task_list.append(
                    (
                        RandomCrop3D(
                            self.args.crop_size,
                            self.args.crop_cat_max,
                            self.args.ignore_index,
                        ),
                        os.path.join(image_mha_folder, series),
                        os.path.join(label_mha_folder, series),
                        os.path.join(
                            self.args.dest_npz_folder, series.replace(".mha", "")
                        ),
                    )
                )

        return task_list


class SemiSupervisedMhaCropper3D(PreCropper3D):
    """Use this class when there are some sample with no annoatations."""

    def parse_task(self):
        task_list = []
        image_mha_folder = os.path.join(self.args.source_mha_folder, "image")
        label_mha_folder = os.path.join(self.args.source_mha_folder, "label")

        for series in os.listdir(image_mha_folder):
            if series.endswith(".mha"):
                image_mha_path = os.path.join(image_mha_folder, series)
                label_mha_path = os.path.join(label_mha_folder, series)
                task_list.append(
                    (
                        RandomCrop3D(
                            self.args.crop_size,
                            self.args.crop_cat_max,
                            self.args.std_thr,
                            self.args.ignore_index,
                        ),
                        image_mha_path,
                        label_mha_path if os.path.exists(label_mha_path) else None,
                        os.path.join(self.args.dest_npz_folder, series.replace(".mha", ""))
                    )
                )

        return task_list

    @staticmethod
    def main_entry():
        """Entry point for command line interface"""
        cropper = SemiSupervisedMhaCropper3D()
