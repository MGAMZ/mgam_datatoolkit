import os
import pdb

import pydicom
import SimpleITK as sitk


def read_dcm_as_sitk(data_directory: str, need_dcms:bool=True
                     ) -> tuple[list[pydicom.FileDataset]|None, sitk.Image|None]:
    """
    [SimpleITK: Dicom Series Read Modify Write](https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html)
    
    读取DICOM文件并返回SimpleITK格式的图像。
    :param dcm_path: DICOM文件路径
    :return: SimpleITK格式的图像
    """
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory, useSeriesDetails=True)
    if not series_IDs:
        print(f"ERROR: given directory `{data_directory}` does not contain a DICOM series.")
        return None, None
    
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0], useSeriesDetails=True)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.SetOutputPixelType(sitk.sitkInt16)

    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    series_reader.MetaDataDictionaryArrayUpdateOn()
    # series_reader.LoadPrivateTagsOn() # disable private tags
    image3D = series_reader.Execute()

    if need_dcms:
        dcms = [pydicom.dcmread(dcm) for dcm in series_file_names]
    else:
        dcms = None
    
    return dcms, image3D
