import os
import argparse
import re
import json
import pdb
import multiprocessing
from tqdm import tqdm
from collections.abc import Sequence

import pydicom
import nrrd
import numpy as np
import SimpleITK as sitk

from mgamdata.io.dcm_toolkit import read_dcm_as_sitk
from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size



def nrrd_to_ItkLabel(
    dcms: list[pydicom.FileDataset], itk_images: sitk.Image, nrrd_path: str
):
    # 获取 DICOM 切片的 Z 轴坐标
    dicom_z_positions = []
    for ds in dcms:
        z = float(ds.ImagePositionPatient[2])
        dicom_z_positions.append(z)
    dicom_z_positions = np.array(dicom_z_positions)

    # 读取 NRRD 文件，获取原点和间距
    data, header = nrrd.read(nrrd_path)
    nrrd_origin = np.array(header.get("space origin", [0.0, 0.0, 0.0]))
    nrrd_spacing = np.array(header.get("spacing", [1.0, 1.0, 1.0]))
    nrrd_size = data.shape

    # 计算 NRRD 切片的 Z 轴坐标
    nrrd_z_positions = nrrd_origin[2] + np.arange(nrrd_size[2]) * nrrd_spacing[2]

    # 找出对应的切片索引
    for idx, z in enumerate(nrrd_z_positions):
        # 寻找最接近的 DICOM 切片
        dicom_idx = np.argmin(np.abs(dicom_z_positions - z))
        break
    else:
        raise ValueError(f"No corresponding DICOM slice found for {nrrd_path}.")

    image_array = sitk.GetArrayFromImage(itk_images)
    mask_array = np.zeros_like(image_array)
    mask_array[dicom_idx] = data.squeeze().transpose(1, 0)
    itk_mask = sitk.GetImageFromArray(mask_array)
    itk_mask.CopyInformation(itk_images)

    return itk_mask


def convert_one_case(args):
    dcms, series_output_folder, spacing, size = args
    output_image_folder = os.path.join(series_output_folder, "image")
    output_label_folder = os.path.join(series_output_folder, "label")
    label_path = os.path.join(os.path.dirname(dcms[0]).replace("img", "label"), "mask.nrrd")

    # 构建路径，保持文件存储结构不变
    # 按照SeriesUID存储，以及自动跳过
    series_id = pydicom.dcmread(dcms[3]).SeriesInstanceUID
    output_image_mha_path = os.path.join(output_image_folder, f"{series_id}.mha")
    output_label_mha_path = os.path.join(output_label_folder, f"{series_id}.mha")
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    if os.path.exists(output_image_mha_path) and os.path.exists(output_label_mha_path):
        tqdm.write(f"Skip {series_id}, already exists.")
        return

    # 读取
    try:
        failed = None
        input_image_dcm, input_image_mha = read_dcm_as_sitk(dcms[0])
        input_label_mha = nrrd_to_ItkLabel(input_image_dcm, input_image_mha, label_path)
    except Exception as e:
        failed = {
            'failed_seriesUID': series_id, 
            'failed_dcm': dcms[0],
            'failed_nrrd': label_path,
            'reason': e}
        input_label_mha = None

    # 重采样
    if spacing is not None:
        assert size is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_spacing(input_image_mha, spacing, "image")
        if input_label_mha is not None:
            input_label_mha = sitk_resample_to_spacing(input_image_mha, spacing, "label")
    if size is not None:
        assert spacing is None, "Cannot set both spacing and size."
        input_image_mha = sitk_resample_to_size(input_image_mha, size, "image")
        if input_label_mha is not None:
            input_label_mha = sitk_resample_to_size(input_image_mha, size, "label")

    # 写入
    sitk.WriteImage(input_image_mha, output_image_mha_path, useCompression=True)
    if input_label_mha is not None:
        sitk.WriteImage(input_label_mha, output_label_mha_path, useCompression=True)

    return True if failed is None else failed


def convert_and_save_nii_to_mha(
    input_dir: str,
    output_dir: str,
    use_mp: bool,
    spacing: Sequence[float | int] | None = None,
    size: Sequence[float | int] | None = None,
):
    task_list = []
    for roots, dirs, files in tqdm(os.walk(input_dir), desc="Scanning", dynamic_ncols=True):
        for file in files:
            if file.endswith(".dcm"):
                dcms_under_folder = [
                    os.path.join(roots, i)
                    for i in os.listdir(roots)
                    if i.endswith(".dcm")
                ]
                sorted_dcms = sorted(
                    dcms_under_folder,
                    key=lambda x: int(re.search(r"(\d+)\.dcm$", x).group(1)),
                )
                task_list.append((sorted_dcms, output_dir, spacing, size))
                break

    exceptions = []
    if use_mp:
        with multiprocessing.Pool() as pool:
            for result in tqdm(
                pool.imap_unordered(convert_one_case, task_list),
                total=len(task_list),
                desc="dcm2mha",
                leave=False,
                dynamic_ncols=True,
            ):
                if result is not True:
                    exceptions.append(result)
    else:
        for args in tqdm(task_list, leave=False, dynamic_ncols=True, desc="dcm2mha"):
            result = convert_one_case(args)
            if result is not True:
                exceptions.append(result)

    return exceptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Containing dcm files.")
    parser.add_argument("output_dir", type=str, help="Save MHA files.")
    parser.add_argument("--mp", action="store_true", help="Use multiprocessing.")
    parser.add_argument("--spacing", type=float, nargs=3, default=None, help="Resample to this spacing.")
    parser.add_argument("--size", type=int, nargs=3, default=None, help="Crop to this size.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "failed.json"), 'w'), indent=4)
    exceptions = convert_and_save_nii_to_mha(args.input_dir, args.output_dir, args.mp, args.spacing, args.size)
    print(f"Failed: {len(exceptions)}")



if __name__ == "__main__":
    main()
