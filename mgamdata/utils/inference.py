import os
import os.path as osp
from pathlib import Path
import pdb
import warnings
from pprint import pprint

import SimpleITK as sitk

from mgamdata.mm.inference import Inferencer_2D

warnings.filterwarnings("ignore", module="pydicom", category=UserWarning)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Inferencer')
    parser.add_argument('cfg_path', type=str, help='Config file path')
    parser.add_argument('ckpt_path', type=str, help='Checkpoint file path')
    parser.add_argument('folder', type=str, help='folder path')
    parser.add_argument('output_folder', type=str, help='Output folder path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print("Initializing Inferencer...")
    inferencer = Inferencer_2D(args.cfg_path, args.ckpt_path)
    pred_generator = inferencer.Inference_FromITKFolder(args.folder)
    
    os.makedirs(args.output_folder, exist_ok=True)
    for itk_image, itk_pred, image_path in pred_generator:
        series_UID = Path(image_path).stem
        sitk.WriteImage(itk_pred, 
                        osp.join(args.output_folder, f"{series_UID}.mha"), 
                        useCompression=True)
    
    print("Inferencing Finished. ITK MHA files saved to: ", args.output_folder)
