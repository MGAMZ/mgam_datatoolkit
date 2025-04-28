import os
import argparse
from multiprocessing import Pool

import SimpleITK as sitk
from tqdm import tqdm

from mgamdata.io.sitk_toolkit import sitk_resample_to_spacing, sitk_resample_to_size

"""
Convert a mha folder with image and label subfolder
into a new mha folder with the same folder structure, 
but volumes are resampled according to either spacing or size.
"""

def process_file(args):
    source_subfolder, target_subfolder, file, spacing, size, field = args
    source_path = os.path.join(source_subfolder, file)
    target_path = os.path.join(target_subfolder, file)
    
    # Load the MHA file
    mha = sitk.ReadImage(source_path)
    
    # Resample the MHA file
    if spacing:
        resampled_mha = sitk_resample_to_spacing(mha, spacing, field)
    elif size:
        resampled_mha = sitk_resample_to_size(mha, size, field)
    else:
        raise ValueError("Either spacing or size must be provided.")
    
    # Save the resampled MHA file
    sitk.WriteImage(resampled_mha, target_path, useCompression=True)
    tqdm.write(f"Resampled {source_path} to {target_path}")


def ResampleFolder(source_folder: str, 
                  target_folder: str, 
                  spacing: list[float] | None = None, 
                  size: list[int] | None = None,
                  use_mp: bool = False):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    subfolders = ['image', 'label']
    
    tasks = []
    
    for subfolder in subfolders:
        source_subfolder = os.path.join(source_folder, subfolder)
        target_subfolder = os.path.join(target_folder, subfolder)
        
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)
        
        files = [f for f in os.listdir(source_subfolder) if f.endswith('.mha')]
        
        for file in files:
            tasks.append((source_subfolder, target_subfolder, file, spacing, size, subfolder))
    
    if use_mp:
        with Pool() as pool:
            list(tqdm(pool.imap(process_file, tasks), 
                      total=len(tasks), 
                      desc="Processing"))
    else:
        for task in tqdm(tasks, desc="Processing"):
            process_file(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample MHA files in a folder.")
    parser.add_argument("source_folder", type=str, help="Path to the source folder containing 'image' and 'label' subfolders.")
    parser.add_argument("target_folder", type=str, help="Path to the target folder to save resampled MHA files.")
    parser.add_argument("--spacing", type=float, nargs='+', default=None, help="Target spacing for resampling.")
    parser.add_argument("--size", type=int, nargs='+', default=None, help="Target size for resampling.")
    parser.add_argument("--mp", action="store_true", default=False, help="Enable multiprocessing.")
    args = parser.parse_args()
    
    ResampleFolder(args.source_folder, args.target_folder, args.spacing, args.size, args.mp)