from abc import abstractmethod
import os
import os.path as osp
import pdb
from tqdm import tqdm
from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from torch import Tensor
from torch.nn import functional as F

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


    def Inference_FromITKFolder(self, folder:str, check_exist_path:str|None=None):
        mha_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.mha'):
                    if check_exist_path is not None:
                        if os.path.exists(osp.join(check_exist_path, file)):
                            print(f"Already inferenced: {file}.")
                            continue
                    else:
                        mha_files.append(osp.join(root, file))
        
        print(f"\nInferencing from Folder: {folder}, Total {len(mha_files)} mha files.\n")
        
        for mha_path in tqdm(mha_files,
                             desc='Inference_FromITKFolder',
                             leave=False,
                             dynamic_ncols=True):
            itk_image = sitk.ReadImage(mha_path)
            itk_image, itk_pred = self.Inference_FromITK(itk_image)
            tqdm.write(f"Successfully inferenced: {os.path.basename(mha_path)}.")
            yield itk_image, itk_pred, mha_path



class Inference_exported(Inferencer_2D):
    @abstractmethod
    def __init__(self, wl, ww):
        ...


    def _set_window(self, inputs):
        inputs = np.clip(inputs, self.wl-self.ww//2, self.wl+self.ww//2)
        inputs = inputs - inputs.min()
        inputs = inputs / inputs.max()
        return inputs.astype(np.float32)


    @torch.inference_mode()
    def Inference_FromNDArray(self, image_array):
        results = []
        for array in tqdm(
                image_array,
                desc="Inferencing",
                total=len(image_array),
                dynamic_ncols=True,
                leave=False,
                mininterval=1):
            result = self.inference(array)
            results.append(result)

        pred = torch.cat(results, axis=0).transpose(0,1)
        return pred # [Class, D, H, W]


    def slide_inference(self, inputs: Tensor) -> Tensor:
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.forward(crop_img)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits


    def whole_inference(self, inputs: Tensor) -> Tensor:
        seg_logits = self.forward(inputs)
        return seg_logits


    def inference(self, inputs: np.ndarray) -> Tensor:
        if self.inference_mode == 'slide':
            seg_logit = self.slide_inference(inputs)
        elif self.inference_mode == 'whole':
            seg_logit = self.whole_inference(inputs)
        else:
            raise ValueError(
                f'Invalid inference mode {self.inference_mode}.'
                'Available options are "slide" and "whole".')

        return seg_logit

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> Tensor:
        ...



class Inference_ONNX(Inference_exported):
    def __init__(self, onnx_path, inference_mode:str='whole', wl=40, ww=400):
        import onnxruntime as ort
        self.model = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.inference_mode = inference_mode
        self.wl = wl
        self.ww = ww
    
    def forward(self, inputs: np.ndarray) -> Tensor:
        inputs = self._set_window(inputs)[None, None]
        assert inputs.ndim == 4
        
        result = self.model.run(['OUTPUT__0'], {'INPUT__0': inputs}) # [1,1,5,H,W]
        result = np.array(result).squeeze()[None]
        result = torch.from_numpy(result)
        return result
