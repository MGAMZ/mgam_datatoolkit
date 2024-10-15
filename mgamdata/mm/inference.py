import os
import os.path as osp
import pdb
from tqdm import tqdm
from typing import Tuple

import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk

from mmseg.models.segmentors import BaseSegmentor
from mmseg.apis.inference import init_model, _preprare_data

from ..io.sitk_toolkit import LoadDcmAsSitkImage




INFERENCER_WORK_DIR = "/fileser51/zhangyiqin.sx/mmseg/work_dirs_inferencer/"



def visualize(image_array, gt_class_idx, pred_class_idx):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image_array, cmap='gray')
    ax[0].set_title("Image")
    ax[1].imshow(gt_class_idx, cmap='tab20')
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_class_idx, cmap='tab20')
    ax[2].set_title("Prediction")
    return fig



class Inferencer_2D:
    def __init__(self, cfg_path, ckpt_path):
        self.model:BaseSegmentor = init_model(cfg_path, ckpt_path)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def Inference_FromNDArray(self, image_array):
        image_array = [i for i in image_array]
        data, is_batch = _preprare_data(image_array, self.model)

        # forward the model
        results = []
        data = self.model.data_preprocessor(data, False)
        inputs = torch.stack(data['inputs'])
        data_samples = [sample.to_dict() for sample in data['data_samples']]
        for array, sample in tqdm(zip(inputs, data_samples),
                                  desc="Inferencing",
                                  total=len(inputs),
                                  dynamic_ncols=True,
                                  leave=False,
                                  mininterval=1):
            result:torch.Tensor = self.model.inference(array[None], [sample])
            results.append(result)

        pred = torch.cat(results, dim=0).transpose(0,1) # [Class, D, H, W]
        return pred


    def Inference_FromITK(self, itk_image:sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
        image_array = sitk.GetArrayFromImage(itk_image) # [D, H, W]
        pred = self.Inference_FromNDArray(image_array) # [Class, D, H, W]
        # 后处理
        pred = pred.argmax(dim=0).to(dtype=torch.uint8, device='cpu').numpy() # [D, H, W]
        itk_pred = sitk.GetImageFromArray(pred)
        itk_pred.CopyInformation(itk_image)
        return itk_image, itk_pred


    def Inference_FromDcm(self, dcm_slide_folder:str, spacing=None):
        image, _, _, _ = LoadDcmAsSitkImage('engineering', dcm_slide_folder, spacing=spacing)
        return self.Inference_FromITK(image)


    def Inference_FromITKFolder(self, folder:str):
        mha_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.mha'):
                    mha_files.append(osp.join(root, file))
        print(f"Inferencing from Folder: {folder}.")
        
        for mha_path in tqdm(mha_files,
                             desc='Inference_FromITKFolder',
                             leave=False,
                             dynamic_ncols=True):
            itk_image = sitk.ReadImage(mha_path)
            itk_image, itk_pred = self.Inference_FromITK(itk_image)
            yield itk_image, itk_pred, mha_path
