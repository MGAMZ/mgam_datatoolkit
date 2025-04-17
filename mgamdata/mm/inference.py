import os
import pdb
from tqdm import tqdm
from collections import defaultdict
from abc import abstractmethod

import torch
import numpy as np
import SimpleITK as sitk
from torch import Tensor

from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from mmseg.apis.inference import _preprare_data

from ..io.sitk_toolkit import LoadDcmAsSitkImage, sitk_resample_to_size, sitk_resample_to_spacing


INFERENCER_WORK_DIR = "/fileser51/zhangyiqin.sx/mmseg/work_dirs_inferencer/"


class Inferencer:
    def __init__(self, cfg_path, ckpt_path, allow_tqdm:bool=True):
        self.allow_tqdm = allow_tqdm
        cfg = Config.fromfile(cfg_path)
        self.model = MODELS.build(cfg.model)
        load_checkpoint(self.model, ckpt_path, map_location='cpu')
        self.pipeline = Compose(cfg.test_pipeline)
        self.model.eval()
        self.model.cuda()
        self.model.requires_grad_(False)

    @abstractmethod
    @torch.inference_mode()
    def Inference_FromNDArray(self, image_array:np.ndarray) -> Tensor:
        ...
    
    def _preprocess(self, imgs:np.ndarray|list[np.ndarray]) -> tuple[Tensor, dict]:
        is_batch = True
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
            is_batch = False

        data = defaultdict(list)
        for img in imgs:
            initial_dict = {
                'img': img,
                'img_shape': img.shape,
                'ori_shape': img.shape,
            }
            data_:dict[str, Tensor|dict] = self.pipeline(initial_dict)
            data['inputs'].append(data_['inputs'])
            data['data_samples'].append(data_['data_samples'])

        return data, is_batch


class SegInferencer(Inferencer):
    def Inference_FromITK(self, itk_image:sitk.Image) -> tuple[sitk.Image, sitk.Image]:
        image_array = sitk.GetArrayFromImage(itk_image) # [Z, Y, X]
        pred = self.Inference_FromNDArray(image_array) # [Class, Z, Y, X]
        # 后处理
        pred = pred.argmax(dim=0).to(dtype=torch.uint8, device='cpu').numpy() # [Z, Y, X]
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
                        if os.path.exists(os.path.join(check_exist_path, file)):
                            print(f"Already inferenced: {file}.")
                            continue
                    mha_files.append(os.path.join(root, file))
        
        print(f"\nInferencing from Folder: {folder}, Total {len(mha_files)} mha files.\n")
        
        for mha_path in tqdm(sorted(mha_files),
                             desc='Inference_FromITKFolder',
                             leave=False,
                             dynamic_ncols=True,
                             disable=not self.allow_tqdm):
            itk_image = sitk.ReadImage(mha_path)
            itk_image, itk_pred = self.Inference_FromITK(itk_image)
            tqdm.write(f"Successfully inferenced: {os.path.basename(mha_path)}.")
            yield itk_image, itk_pred, mha_path


class Inferencer_2D(SegInferencer):
    @torch.inference_mode()
    def Inference_FromNDArray(self, image_array:np.ndarray) -> Tensor:
        assert image_array.ndim == 3, "Input image must be 3D, got: {}.".format(image_array.shape)
        image_array = [i for i in image_array]
        data, is_batch = self._preprocess(image_array)

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
                                  mininterval=1,
                                  disable=not self.allow_tqdm):
            result:torch.Tensor = self.model.inference(array[None], [sample])
            results.append(result)

        pred = torch.cat(results, dim=0).transpose(0,1) # [Class, D, H, W]
        return pred


class Inference_ONNX(Inferencer_2D):
    def __init__(self, onnx_path):
        import onnxruntime as ort # type: ignore
        self.model = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    @torch.inference_mode()
    def Inference_FromNDArray(self, image_array):
        results = []
        for array in tqdm(
                image_array,
                desc="Inferencing",
                total=len(image_array),
                dynamic_ncols=True,
                leave=False,
                mininterval=1,
                disable=not self.allow_tqdm):
            result = self.inference(array)
            results.append(result)
        pred = torch.cat(results, axis=0).transpose(0,1)
        return pred # [Class, D, H, W]

    def forward(self, inputs: np.ndarray) -> Tensor:
        inputs = self._set_window(inputs)[None, None]
        assert inputs.ndim == 4
        result = self.model.run(['OUTPUT__0'], {'INPUT__0': inputs}) # [1,1,5,H,W]
        result = np.array(result).squeeze()[None]
        result = torch.from_numpy(result)
        return result


class Inferencer_3D(SegInferencer):
    def __init__(self, spacings=[None,None,None], sizes=[None,None,None], *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(spacings) == 3, "Spacings must be a list of 3 elements, got: {}.".format(spacings)
        assert len(sizes) == 3, "Sizes must be a list of 3 elements, got: {}.".format(sizes)
        assert not any([spacing is not None and size is not None 
                        for spacing, size in zip(spacings, sizes)]), \
            "Can not specify spacing and size for one dimension at the same time, got spacings: {}, sizes: {}.".format(spacings, sizes)
        self.spacings = spacings
        self.sizes = sizes
    
    @torch.inference_mode()
    def Inference_FromNDArray(self, image_array) -> Tensor:
        data, is_batch = _preprare_data(image_array, self.model)
        data = self.model.data_preprocessor(data, False)
        with torch.autocast('cuda'):
            img_input = data['inputs'][0][None]
            img_meta = [d.to_dict() for d in data['data_samples']]
            seg_result = self.model.inference(img_input, img_meta)
            return seg_result.squeeze(0) # [N, Class, Z, Y, X] -> [Class, Z, Y, X]

    def Inference_FromITK(self, itk_image:sitk.Image) -> tuple[sitk.Image, sitk.Image]:
        if any(self.spacings):
            # the dimension order aligns to Z Y X.
            # complementation on each dimension
            ori_spacing = itk_image.GetSpacing()[::-1]
            ori_size = itk_image.GetSize()[::-1]
            target_spacing = [target or ori for ori, target in zip(ori_spacing, self.spacings)]
            target_size = [target or ori for ori, target in zip(ori_size, self.sizes)]
            # resampling
            if any(self.spacings):
                itk_image = sitk_resample_to_spacing(itk_image, target_spacing, "image")
            if any(self.sizes):
                itk_image = sitk_resample_to_size(itk_image, target_size, "image")
            # inference
            return super().Inference_FromITK(itk_image)