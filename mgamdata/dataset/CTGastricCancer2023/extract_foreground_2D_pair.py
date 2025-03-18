import os
import argparse
import pdb
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

import cv2
import numpy as np
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser(description='Extract 2D slices with foreground from 3D image-label pairs')
    parser.add_argument('input_dir', type=str, help='Input directory containing image and label folders')
    parser.add_argument('output_dir', type=str, help='Output directory for extracted 2D slices')
    parser.add_argument('--mp', action='store_true', help='Enable multiprocessing')
    parser.add_argument('--num-workers', type=int, default=None)
    return parser.parse_args()


def find_foreground_slice(label_array):
    """Find the slice that contains foreground (label > 0)"""
    # For each slice along z-axis, check if it contains any foreground
    for z in range(label_array.shape[0]):
        if np.any(label_array[z] > 0):
            return z
    return None


def process_file(label_file, input_dir, output_dir):
    """Process a single image-label pair and save the foreground slice"""
    label_dir = os.path.join(input_dir, 'label')
    image_dir = os.path.join(input_dir, 'image')
    
    if not os.path.exists(os.path.join(image_dir, label_file)):
        print(f"Warning: No matching image found for {label_file}")
        return
    
    # Load label and image
    label_path = os.path.join(label_dir, label_file)
    image_path = os.path.join(image_dir, label_file)
    
    try:
        label_sitk = sitk.ReadImage(label_path)
        image_sitk = sitk.ReadImage(image_path)
        label_array = sitk.GetArrayFromImage(label_sitk).astype(np.uint8)
        image_array = sitk.GetArrayFromImage(image_sitk).astype(np.int16)
        
        # Find slice with foreground
        foreground_slice_idx = find_foreground_slice(label_array)
        if foreground_slice_idx is None:
            print(f"Warning: No foreground found in {label_file}")
            return
        
        label_slice = label_array[foreground_slice_idx]
        image_slice = image_array[foreground_slice_idx]
        
        # save as tiff
        output_basename = os.path.splitext(label_file)[0]
        cv2.imwrite(os.path.join(output_dir, 'label', output_basename + '.tiff'), label_slice)
        cv2.imwrite(os.path.join(output_dir, 'image', output_basename + '.tiff'), image_slice)
        
        return label_file  # Return filename to indicate success
    except Exception as e:
        print(f"Error processing {label_file}: {e}")
        return None


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'label'), exist_ok=True)
    
    label_dir = os.path.join(args.input_dir, 'label')
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.mha')]
    
    # Set number of workers
    num_workers = args.num_workers or mp.cpu_count()
    
    if args.mp and len(label_files) > 1:
        print(f"Running with {num_workers} parallel processes")
        # Create a partial function with fixed arguments
        process_func = partial(process_file, 
                              input_dir=args.input_dir, 
                              output_dir=args.output_dir)
        
        # Run processing in parallel
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, label_files),
                total=len(label_files),
                desc='Processing files',
                dynamic_ncols=True
            ))
            
        # Count successful files
        successful = [r for r in results if r is not None]
        print(f"Successfully processed {len(successful)} out of {len(label_files)} files")
    else:
        # Process files sequentially
        successful = []
        for label_file in tqdm(label_files, desc='Processing files', dynamic_ncols=True):
            result = process_file(label_file, args.input_dir, args.output_dir)
            if result is not None:
                successful.append(result)
        
        print(f"Successfully processed {len(successful)} out of {len(label_files)} files")


if __name__ == '__main__':
    main()