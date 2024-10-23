import argparse
import os
import pdb

import cv2
import torch
import numpy as np
import SimpleITK as sitk

from mmseg.apis.inference import init_model, _preprare_data
from mmseg.models.segmentors import EncoderDecoder




class mm_model_warpper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model:EncoderDecoder = init_model(args.cfg_path, args.ckpt_path).cuda()
        self.requires_grad_(False)
        self.eval()
    
    def forward(self, inputs):
        return self.model._forward(inputs)



def post_process(no_jit_pred, jit_pred, gt, save_dir):
    no_jit_pred = no_jit_pred.argmax(1).cpu().numpy().astype(np.uint8)
    jit_pred = jit_pred.argmax(1).cpu().numpy().astype(np.uint8)
    inconsistent = (no_jit_pred!=jit_pred).sum()
    
    no_jit_pred = sitk.GetImageFromArray(no_jit_pred[0])
    jit_pred = sitk.GetImageFromArray(jit_pred[0])
    gt = sitk.GetImageFromArray(gt[0])
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



def export_jit(model, image:torch.Tensor, label:np.ndarray):
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
    
    return output



def export_onnx(model, image:torch.Tensor, label:np.ndarray):
    torch.onnx.export(
        model, 
        image, 
        args.save_path, 
        verbose=False, 
        opset_version=11, 
        input_names=['INPUT__0'], 
        output_names=['OUTPUT__0'],
        dynamic_axes={
            'INPUT__0': {0: "batch_size"}, 
            'OUTPUT__0': {0: "batch_size"}}
    )
    
    import onnxruntime as ort
    # output: torch.Size([1, 5, 512, 512]) | torch.float32
    direct_pred = model(image)
    
    # load onnx model and test it
    ort_session = ort.InferenceSession(args.save_path)
    
    out1:np.ndarray = ort_session.run(['OUTPUT__0'], {'INPUT__0': image.cpu().numpy()})[0]
    print(f"The output has shape: {out1.shape} dtype: {out1.dtype}")
    out1 = torch.from_numpy(out1)
    inconsistent = post_process(direct_pred, out1, label, os.path.dirname(args.save_path))
    print(f"There {inconsistent} inconsistent pixels.")
    
    out2:np.ndarray = ort_session.run(['OUTPUT__0'], {'INPUT__0': image.repeat(4, 1, 1, 1).cpu().numpy()})[0]
    print(f"The output has shape: {out2.shape} dtype: {out2.dtype}")
    out2 = torch.from_numpy(out2)
    inconsistent = post_process(direct_pred, out2, label[None].repeat(4,0), os.path.dirname(args.save_path))
    print(f"There {inconsistent} inconsistent pixels.")
    
    return out1



def main(args):
    model = mm_model_warpper(args)
    
    # 拿一个样本来，简单预处理
    image, label = fetch_sample()
    label = label[None]
    data, _ = _preprare_data(image, model.model)
    
    # 处理后，神经网络的输入如下
    # [1, 1, 512, 512]
    image = torch.stack(data['inputs']).to(dtype=torch.float32, device='cuda')
    print(f"The Input has shape: {image.shape}")

    if args.export_type == 'onnx':
        export_onnx(model, image, label)
    elif args.export_type == 'jit':
        export_jit(model, image, label)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export model for deployment')
    parser.add_argument('export_type', type=str, choices=['onnx', 'jit'], help='Export type')
    parser.add_argument('cfg_path', type=str, help='Config file path')
    parser.add_argument('ckpt_path', type=str, help='Checkpoint file path')
    parser.add_argument('save_path', type=str, help='Output path')
    args = parser.parse_args()
    main(args)
