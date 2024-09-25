import argparse
import os
import pdb

import cv2
import torch
import numpy as np
import SimpleITK as sitk

from mmseg.apis.inference import init_model, _preprare_data
from mmseg.models.segmentors import EncoderDecoder


def parse_args():
    parser = argparse.ArgumentParser(description='Jit trace and export')
    parser.add_argument('cfg_path', type=str, help='Config file path')
    parser.add_argument('ckpt_path', type=str, help='Checkpoint file path')
    parser.add_argument('save_path', type=str, help='Output path')
    return parser.parse_args()


class mm_model_warpper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model:EncoderDecoder = init_model(args.cfg_path, args.ckpt_path).cuda()
        self.requires_grad_(False)
        self.eval()
    
    def forward(self, inputs):
        return self.model._forward(inputs)


def post_process(no_jit_pred, jit_pred, gt, save_dir):
    no_jit_pred = no_jit_pred.squeeze().cpu().numpy().argmax(0).astype(np.uint8)
    jit_pred = jit_pred.squeeze().cpu().numpy().argmax(0).astype(np.uint8)
    inconsistent = (no_jit_pred!=jit_pred).sum()
    
    no_jit_pred = sitk.GetImageFromArray(no_jit_pred)
    jit_pred = sitk.GetImageFromArray(jit_pred)
    gt = sitk.GetImageFromArray(gt)
    sitk.WriteImage(no_jit_pred, os.path.join(save_dir, "no_jit.mha"), useCompression=True)
    sitk.WriteImage(jit_pred, os.path.join(save_dir, "jit.mha"), useCompression=True)
    sitk.WriteImage(gt, os.path.join(save_dir, "gt.mha"), useCompression=True)
    
    return inconsistent


def fetch_sample():
    image_path = "/fileser51/zhangyiqin.sx/Sarcopenia_Data/Batch5_7986/ForegroundTIFF_original_EngineerSort/image/1.2.156.112605.66988329457737.240423090040.3.14672.101381/1.2.156.112605.66988329457737.240423090040.3.14672.101381_0.tiff"
    label_path = image_path.replace('image', 'label')
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.int16)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    
    return image, label


if __name__ == '__main__':
    args = parse_args()
    model = mm_model_warpper(args)
    
    # 拿一个样本来，简单预处理
    image, label = fetch_sample()
    data, _ = _preprare_data(image, model.model)
    
    # 处理后，神经网络的输入如下
    # [1, 1, 512, 512]
    image = torch.stack(data['inputs']).to(dtype=torch.float32, device='cuda')
    print(f"The Input has shape: {image.shape}")
    
    # JIT trace
    exported = torch.jit.trace(model, image)
    
    # output: torch.Size([1, 5, 512, 512]) | torch.float32
    # 直接推理
    no_jit_pred = model(image)
    # JIT推理
    output:torch.Tensor = exported(image)
    
    # log
    print(f"The output has shape: {output.shape} dtype: {output.dtype}")
    exported.save(args.save_path)
    inconsistent = post_process(no_jit_pred, output, label, os.path.dirname(args.save_path))
    print(f"There {inconsistent} inconsistent pixels.")