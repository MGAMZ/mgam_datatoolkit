import os
import os.path as osp
import pdb
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
from pprint import pprint
from colorama import Fore, Style

import torch
import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import SimpleITK as sitk

from mmengine.logging import print_log
from mmengine.config import Config
from mmseg.models.segmentors import BaseSegmentor
from mmseg.apis.inference import init_model, _preprare_data

from .mmeng_PlugIn import DynamicRunnerSelection
from ..io.sitk_toolkit import LoadDcmAsSitkImage





INFERENCER_WORK_DIR = "/fileser51/zhangyiqin.sx/mmseg/work_dirs_inferencer/"


def dice_coeff(pred, target):
    eps = 1e-7
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)


def visualize(image_array, gt_class_idx, pred_class_idx):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image_array, cmap='gray')
    ax[0].set_title("Image")
    ax[1].imshow(gt_class_idx, cmap='tab20')
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_class_idx, cmap='tab20')
    ax[2].set_title("Prediction")
    return fig

# 整体的思路是，不管是什么文件，都要转换为Mhz格式，然后统一进行推理。
class Inferencer:
    def __init__(self, cfg_path:str, checkpoint_path:str, mp:bool=False):
        self.cfg = Config.fromfile(cfg_path)
        if mp is False:
            self.cfg.pop('launcher')
        self.cfg.work_dir = INFERENCER_WORK_DIR
        self.checkpoint_path = checkpoint_path
        self.runner = DynamicRunnerSelection(self.cfg)
        self.runner.load_checkpoint(self.checkpoint_path)


class Inferencer_2D:
    def __init__(self, cfg_path, ckpt_path):
        self.model:BaseSegmentor = init_model(cfg_path, ckpt_path)
    
    @torch.inference_mode()
    def Inference_FromDcm(self, dcm_slide_folder:str, spacing=None
            ) -> Tuple[sitk.Image, sitk.Image]:
        image, _, _, _ = LoadDcmAsSitkImage('engineering', dcm_slide_folder, spacing=spacing)
        image_array = sitk.GetArrayFromImage(image)
        image_array = [i for i in image_array]
        data, is_batch = _preprare_data(image_array, self.model)
        
        # forward the model
        results = []
        data = self.model.data_preprocessor(data, False)
        inputs = torch.stack(data['inputs'])
        data_samples = [sample.to_dict() for sample in data['data_samples']]
        for array, sample in tqdm(zip(inputs, data_samples),
                                  desc="Inference",
                                  total=len(inputs),
                                  dynamic_ncols=True,
                                  leave=False):
            result = self.model.inference(array[None], [sample])
            results.append(result)
        
        # 后处理
        pred = torch.cat(results, axis=0).transpose(0,1)   # [Class, D, H, W]
        pred = pred.argmax(axis=0).to(dtype=torch.uint8, device='cpu').numpy()
        pred = sitk.GetImageFromArray(pred)
        pred.CopyInformation(image)
        return image, pred
