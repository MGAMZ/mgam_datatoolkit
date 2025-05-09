import os
import argparse
import datetime
import pdb
import json
import multiprocessing as mp
from abc import abstractmethod
from collections.abc import Sequence
from re import A
from textwrap import indent
from tqdm import tqdm

from ..io.sitk_toolkit import (
    sitk,
    nii_to_sitk,
    sitk_resample_to_spacing,
    sitk_resample_to_size,
    sitk_resample_to_image,
)
from ..io.dcm_toolkit import read_dcm_as_sitk



class StandardFileFormatter:
    def __init__(self) -> None:
        self.args = self.argparse().parse_args()

    @abstractmethod
    def tasks(self) -> list:
        """Return a list of args for `convert_one_sample`."""

    @staticmethod
    def _series_id(image_path: str|None, label_path: str|None) -> str:
        return os.path.basename(label_path).replace(".nii.gz", "")

    def convert_one_sample(self, args):
        image_path, label_path, dest_folder, series_id, spacing, size = args
        convertion_log = {
            "img": os.path.relpath(image_path, self.args.data_root) if image_path is not None else None,
            "ann": os.path.relpath(label_path, self.args.data_root) if label_path is not None else None,
            "id": series_id,
        }

        # source path and output folder
        output_image_folder = os.path.join(dest_folder, "image")
        output_label_folder = os.path.join(dest_folder, "label")
        output_image_mha_path = os.path.join(output_image_folder, f"{series_id}.mha")
        output_label_mha_path = os.path.join(output_label_folder, f"{series_id}.mha")
        os.makedirs(output_image_folder, exist_ok=True)
        os.makedirs(output_label_folder, exist_ok=True)
        if os.path.exists(output_image_mha_path):
            if label_path is None \
            or os.path.exists(output_label_mha_path) \
            or not os.path.exists(label_path):
                return convertion_log

        try:

            if isinstance(image_path, str) and ".dcm" in image_path:
                input_image_mha, input_label_mha = self.convert_one_sample_dcm(image_path, label_path)
            elif ".nii.gz" in image_path:
                input_image_mha, input_label_mha = self.convert_one_sample_nii(image_path, label_path)
            if input_image_mha is None:
                convertion_log["id"] = "error"
                convertion_log["error"] = "No image found."
                return convertion_log

            # resample
            if spacing is not None:
                assert size is None, "Cannot set both spacing and size."
                input_image_mha = sitk_resample_to_spacing(input_image_mha, spacing, "image")
                if not isinstance(input_image_mha, sitk.Image):
                    convertion_log["id"] = "error"
                    convertion_log["error"] = "Resample to spacing failed."
                    convertion_log["resample_error_detail"] = input_image_mha
                    return convertion_log
            elif size is not None:
                assert spacing is None, "Cannot set both spacing and size."
                input_image_mha = sitk_resample_to_size(input_image_mha, size, "image")

            # Align label to image, if label exists.
            if input_label_mha is not None and os.path.exists(label_path):
                input_label_mha = sitk_resample_to_image(input_label_mha, input_image_mha, "label")

            input_image_mha = sitk.DICOMOrient(input_image_mha, 'LPI')
            sitk.WriteImage(input_image_mha, output_image_mha_path, useCompression=True)
            if input_label_mha is not None and os.path.exists(label_path):
                assert (input_image_mha.GetSize() == input_label_mha.GetSize()), \
                    f"Image {input_image_mha.GetSize()} and label {input_label_mha.GetSize()} size mismatch."
                input_label_mha = sitk.DICOMOrient(input_label_mha, 'LPI')
                sitk.WriteImage(input_label_mha, output_label_mha_path, useCompression=True)
        
        except Exception as e:
            convertion_log["id"] = "error"
            convertion_log["error"] = str(e)
            error_info = f"SeriesUID{series_id} | " + str(e)
            convertion_log["Unknown_error_detail"] = error_info
            print(error_info)
        
        return convertion_log
    
    @staticmethod
    def convert_one_sample_dcm(image_path:str, label_path:str):
        input_image_dcms, input_image_mha = read_dcm_as_sitk(image_path)
        return input_image_mha, None
    
    @staticmethod
    def convert_one_sample_nii(image_path, label_path):
        if image_path is not None and os.path.exists(image_path):
            input_image_mha = nii_to_sitk(image_path, "image")
        else:
            input_image_mha = None
        if label_path is not None and os.path.exists(label_path):
            input_label_mha = nii_to_sitk(label_path, "label")
        else:
            input_label_mha = None
        return input_image_mha, input_label_mha

    def execute(self):
        task_list = self.tasks()
        per_sample_log = []

        if self.args.mp:
            with mp.Pool() as pool:
                for result in tqdm(
                    pool.imap_unordered(self.convert_one_sample, task_list),
                    total=len(task_list),
                    desc="convert2mha",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    per_sample_log.append(result)
        else:
            for args in tqdm(
                task_list, leave=False, dynamic_ncols=True, desc="convert2mha"
            ):
                result = self.convert_one_sample(args)
                per_sample_log.append(result)
        
        convertion_log = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_root": self.args.data_root, 
            "dest_root": self.args.dest_root,
            "spacing": self.args.spacing,
            "size": self.args.size,
            "per_sample_log": per_sample_log,
        }
        json.dump(convertion_log, open(os.path.join(self.args.dest_root, "convertion_log.json"), "w"), indent=4)
        print(f"Converted {len(per_sample_log)} series. Saved to {self.args.dest_root}.")

    def argparse(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Convert all NIfTI files in a directory to MHA format.")
        parser.add_argument("data_root", type=str, help="Containing NIfTI files.")
        parser.add_argument("dest_root", type=str, help="Save MHA files.")
        parser.add_argument("--mp", action="store_true", help="Use multiprocessing.")
        parser.add_argument("--spacing", type=float, nargs=3, default=None, help="Resample to this spacing.")
        parser.add_argument("--size", type=int, nargs=3, default=None, help="Crop to this size.")
        return parser


class format_from_standard(StandardFileFormatter):
    def tasks(self) -> list:
        task_list = []
        image_folder = os.path.join(self.args.data_root, "image")

        for series_name in os.listdir(image_folder):
            if series_name.endswith(".nii.gz"):
                image_path = os.path.join(image_folder, series_name)
                label_path = image_path.replace("image", "label")
                series_id = self._series_id(image_path, label_path)
                task_list.append(
                    (
                        image_path,
                        label_path,
                        self.args.dest_root,
                        series_id,
                        self.args.spacing,
                        self.args.size,
                    )
                )
        return task_list


class format_from_nnUNet(StandardFileFormatter):
    def tasks(self) -> list:
        task_list = []
        image_folder = os.path.join(self.args.data_root, "image")

        for series_name in os.listdir(image_folder):
            if series_name.endswith(".nii.gz"):
                image_path = os.path.join(image_folder, series_name)
                label_path = image_path.replace("image", "label").replace(
                    "_0000.nii.gz", ".nii.gz"
                )
                series_id = self._series_id(image_path, label_path)
                task_list.append(
                    (
                        image_path,
                        label_path,
                        self.args.dest_root,
                        series_id,
                        self.args.spacing,
                        self.args.size,
                    )
                )
        return task_list


class format_from_unsup_datasets(StandardFileFormatter):
    MINIMUM_DCM_SLICES = 30

    @staticmethod
    def _series_id(image_path: str|None, label_path: str|None) -> str:
        raise NotImplementedError("Unsup dataset requires no series_id")

    def tasks(self) -> list:
        task_list = []
        id = 0
        deprecated_dcm = 0
        for root, dirs, files in tqdm(os.walk(self.args.data_root), desc="Searching"):
            dcm_files = [f for f in files if f.lower().endswith('.dcm')]
            nii_files = [f for f in files if f.lower().endswith('.nii') or f.lower().endswith('.nii.gz')]

            # dcm files
            if len(dcm_files) >= self.MINIMUM_DCM_SLICES:
                first_dcm = os.path.join(root, dcm_files[0])
                tqdm.write(f"Found available dcm series: {first_dcm}")
                label_path = None
                task_list.append(
                    (
                        first_dcm,
                        label_path,
                        self.args.dest_root,
                        id,
                        self.args.spacing,
                        self.args.size,
                    )
                )
                id += 1
            else:
                deprecated_dcm = 0

            # nii files
            for nii in nii_files:
                nii_path = os.path.join(root, nii)
                tqdm.write(f"Found available nii file: {nii_path}")
                label_path = nii_path.replace("image", "label").replace("_0000.nii.gz", ".nii.gz")
                task_list.append(
                    (
                        nii_path,
                        label_path,
                        self.args.dest_root,
                        id,
                        self.args.spacing,
                        self.args.size,
                    )
                )
                id += 1

        print(f"Total {len(task_list)+deprecated_dcm} series, "
              f"among which {len(task_list)} available series, "
              f"{deprecated_dcm} deprecated dcms series.")
        return task_list
