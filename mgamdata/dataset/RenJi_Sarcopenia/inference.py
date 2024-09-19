import os
import os.path as osp
from pathlib import Path
import pdb
import datetime
import warnings
from tqdm import tqdm
from typing import List, Dict, Tuple
from pprint import pprint
from colorama import Fore, Style

import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from aitrox.utils.inference import Inferencer_2D


warnings.filterwarnings("ignore", module="pydicom", category=UserWarning)

USE_TQDM = True
if USE_TQDM:
    print = tqdm.write


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Inference on DCM')
    parser.add_argument('mode', type=str, default='single', help='Inference mode',)
    parser.add_argument('cfg_path', type=str, help='Config file path',
                        default="/fileser51/zhangyiqin.sx/mmseg/work_dirs/0.8.3.FixRange/round_1/MedNext_3D/MedNext_3D.py")
    parser.add_argument('ckpt_path', type=str, help='Checkpoint file path',
                        default="/fileser51/zhangyiqin.sx/mmseg/work_dirs/0.8.3.FixRange/round_1/MedNext_3D/best_Perf_mDice_iter_24000.pth")
    parser.add_argument('dcm_folder', type=str, help='DCM folder path',
                        default="/fileser51/zhangyiqin.sx/Sarcopenia_Data/batch5_8009_DCM/1.3.12.2.1107.5.1.7.154484.30000024081309123198300000390")
    parser.add_argument('output_folder', type=str, help='Output folder path',
                        default="/fileser51/zhangyiqin.sx/mmseg/")
    return parser.parse_args()


def Inference_One_Series(dcm_folder:str, output_folder:str):
    series_id = os.path.basename(dcm_folder)
    print(f"Inferencing {dcm_folder}")
    image, pred = inferencer.Inference_FromDcm(dcm_folder)
    
    print(f"Inference done, saving to {output_folder}.")
    # compressionLevel 0-9 越高越慢
    sitk.WriteImage(pred, osp.join(output_folder, f'{series_id}.mha'), 
                    useCompression=True, compressionLevel=5)
    print(f"Saved to {output_folder}.")


if __name__ == '__main__':
    args = parse_args()
    
    print("Initializing Inferencer")
    inferencer = Inferencer_2D(args.cfg_path, args.ckpt_path)
    
    if args.mode == 'single':
        Inference_One_Series(args.dcm_folder, args.output_folder)
    
    elif args.mode == 'recursive':
        for series in tqdm(os.listdir(args.dcm_folder),
                           desc='Folder Inference',
                           dynamic_ncols=True,
                           leave=False):
            Inference_One_Series(osp.join(args.dcm_folder, series), args.output_folder)
    
