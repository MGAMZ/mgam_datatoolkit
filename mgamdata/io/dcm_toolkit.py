import os
import pdb

import pydicom
import nrrd
import SimpleITK as sitk


def read_dcm_as_sitk(dcm_path: str) -> tuple[list[pydicom.FileDataset], sitk.Image]:
    """
    读取DICOM文件并返回SimpleITK格式的图像。
    :param dcm_path: DICOM文件路径
    :return: SimpleITK格式的图像
    """
    itk_reader = sitk.ImageSeriesReader()
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(
        os.path.dirname(dcm_path), useSeriesDetails=True
    )
    if not series_ids:
        raise ValueError(f"No DICOM series found in {dcm_path}.")
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        os.path.dirname(dcm_path),
        series_ids[0],
        useSeriesDetails=True,
        recursive=False,
        loadSequences=False,
    )
    itk_reader.SetFileNames(series_file_names)

    dcms = [pydicom.dcmread(dcm) for dcm in series_file_names]
    itk_image = itk_reader.Execute()
    return dcms, itk_image
