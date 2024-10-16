import os
import pdb

import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk




def convert_nii_sitk(nii_path:str) -> sitk.Image:
    nib_img = nib.load(nii_path)
    nib_array = nib_img.get_fdata()
    nib_meta = nib_img.header
    nib_spacing = nib_meta['pixdim'][1:4].tolist()
    nib_origin = nib_meta.get_qform()[0:3, 3].tolist()
    nib_direction = nib_meta.get_qform()[0:3, 0:3].flatten().tolist()
    
    sitk_img = sitk.GetImageFromArray(nib_array)
    sitk_img.SetSpacing(nib_spacing[::-1])
    sitk_img.SetOrigin(nib_origin[::-1])
    sitk_img.SetDirection(nib_direction[::-1])
    return sitk_img



if __name__ == "__main__":
    nii_path = '/fileser51/zhangyiqin.sx/Totalsegmentator_Data/Totalsegmentator_dataset_v201/s0000/ct.nii.gz'
    sitk_img = convert_nii_sitk(nii_path)