import os
import argparse
import json
import multiprocessing as mp
from abc import abstractmethod
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk

from ..process.GeneralPreProcess import RandomCrop3D
from .NDArray import unsafe_astype


class PreCropper3D:
    def __init__(self):
        self.main()

    def arg_parse(self):
        argparser = argparse.ArgumentParser("Pre-Random-Crop 3D")
        argparser.add_argument(
            "source_mha_folder", type=str, help="The folder containing mha files."
        )
        argparser.add_argument(
            "dest_npz_folder", type=str, help="The folder to save npz files."
        )
        argparser.add_argument(
            "--crop-size",
            type=int,
            nargs=3,
            required=True,
            help="The size of cropped volume.",
        )
        argparser.add_argument(
            "--crop-cat-max",
            type=float,
            default=1.0,
            help="Max ratio for single catagory can occupy.",
        )
        argparser.add_argument(
            "--num-cropped-ratio",
            type=int,
            default=4,
            help="The number of cropped volumes per series.",
        )
        argparser.add_argument(
            "--ensure-index",
            type=int,
            default=None,
            nargs="+",
            help="The index to ensure in segmentation.",
        )
        argparser.add_argument(
            "--ensure-ratio",
            type=float,
            default=None,
            help="The chance for an ensurance to perform.",
        )
        argparser.add_argument(
            "--ignore-index",
            type=int,
            default=255,
            help="The index to ignore in segmentation. "
            "It will not taken into consideration during "
            "the determination of whether the cropped patch "
            "meets the `crop-cat-max` setting.",
        )
        argparser.add_argument(
            "--mp",
            action="store_true",
            default=False,
            help="Whether to use multiprocessing.",
        )
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

    def all_index_ensured(self, label: np.ndarray):
        if self.args.ensure_index is None or np.random.rand() > self.args.ensure_ratio:
            return True
        else:
            return any(index not in label for index in self.args.ensure_index)

    def crop_per_series(self, args: tuple) -> dict:
        cropper, image_itk_path, anno_itk_path, save_folder = args
        cropper: RandomCrop3D
        os.makedirs(save_folder, exist_ok=True)
        existed_classes = {}

        for crop_idx, (img_array, anno_array) in enumerate(
            self.Crop3D(cropper, image_itk_path, anno_itk_path)
        ):
            save_path = os.path.join(
                save_folder, f"{os.path.basename(save_folder)}_{crop_idx}.npz"
            )
            np.savez_compressed(
                file=save_path,
                img=img_array,
                gt_seg_map=anno_array if anno_array is not None else np.nan,
            )
            existed_classes[os.path.basename(save_path)] = (
                np.unique(anno_array).tolist() if anno_array is not None else None
            )

        json.dump(
            {
                "series_id": os.path.basename(save_folder),
                "shape": img_array.shape,
                "num_patches": crop_idx + 1,
                "anno_available": anno_array is not None,
                "class_within_patch": existed_classes,
            },
            open(
                os.path.join(save_folder, "SeriesMeta.json"),
                "w",
            ),
            indent=4,
        )

        return {
            os.path.basename(save_folder): {
                "num_patches": crop_idx + 1,
                "anno_available": anno_array is not None,
            }
        }

    def Crop3D(
        self, cropper, image_itk_path: str, anno_itk_path: str | None  # type: ignore
    ):
        from .GeneralPreProcess import RandomCrop3D

        cropper: RandomCrop3D

        image_itk_image = sitk.ReadImage(image_itk_path)
        image_array = sitk.GetArrayFromImage(image_itk_image)
        if anno_itk_path is not None:
            anno_itk_image = sitk.ReadImage(anno_itk_path)
            anno_array = sitk.GetArrayFromImage(anno_itk_image)
        else:
            anno_array = None

        # not giving the seg_fields, RandomCrop3D will not crop label.
        data = {
            "img": image_array,
            "gt_seg_map": anno_array,
            "seg_fields": ["gt_seg_map"] if anno_array is not None else [],
        }

        num_cropped = (
                int(np.prod(np.array(image_array.shape) / np.array(cropper.crop_size)))
                * self.args.num_cropped_ratio
            )

        for i in range(num_cropped):
            # if no label, can't check cat_max_ratio
            if anno_itk_path is None:
                cropper.cat_max_ratio = 1.0

            crop_bbox = cropper.crop_bbox(data)

            if anno_itk_path is not None:
                cropped_anno_array: np.ndarray = cropper.crop(anno_array, crop_bbox)
                cropped_anno_array = unsafe_astype(cropped_anno_array, np.uint8)

                if self.all_index_ensured(cropped_anno_array):
                    cropped_image_array: np.ndarray = cropper.crop(
                        image_array, crop_bbox
                    )
                    cropped_image_array = unsafe_astype(cropped_image_array, np.int16)
                    yield cropped_image_array, cropped_anno_array
                else:
                    tqdm.write(
                        f"deprecated due to failing to ensure index: {anno_itk_path} | crop_idx: {i}"
                    )

            else:
                cropped_image_array: np.ndarray = cropper.crop(image_array, crop_bbox)
                cropped_image_array = unsafe_astype(cropped_image_array, np.int16)
                yield cropped_image_array, None


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
                            self.args.ignore_index,
                        ),
                        image_mha_path,
                        label_mha_path if os.path.exists(label_mha_path) else None,
                        os.path.join(
                            self.args.dest_npz_folder, series.replace(".mha", "")
                        ),
                    )
                )

        return task_list
