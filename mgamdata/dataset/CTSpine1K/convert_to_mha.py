import os
import pdb
import argparse
import warnings
warnings.filterwarnings("ignore")

import SimpleITK as sitk
from tqdm import tqdm



def process_item(task):
    """
    单个数据处理函数，用于多进程/单进程中被调用
    task: (data_file_path, label_file_path, base_name, output_image_path, output_label_path)
    """
    data_file_path, label_file_path, base_name, output_image_path, output_label_path = task
    out_img_path = os.path.join(output_image_path, base_name + '.mha')
    out_label_path = os.path.join(output_label_path, base_name + '.mha')
    if os.path.exists(out_img_path) and os.path.exists(out_label_path):
        print(f"File {base_name} already exists, skipping...")
        return

    reader = sitk.ImageFileReader()
    reader.SetFileName(data_file_path)
    image = reader.Execute()

    sitk.WriteImage(image, out_img_path, useCompression=True)
    label = sitk.ReadImage(label_file_path)
    label.CopyInformation(image)
    sitk.WriteImage(label, out_label_path, useCompression=True)


def convert_ctspine_data(input_dir, out_dir, use_mp=False):
    # 需要遍历的目录
    data_path = os.path.join(input_dir, 'data')
    label_path = os.path.join(input_dir, 'label')
    output_image_path = os.path.join(out_dir, 'image')
    output_label_path = os.path.join(out_dir, 'label')

    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)

    # 获取 data 文件夹下的四个子文件夹
    data_subfolders = [
        os.path.join(data_path, f) for f in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, f))
    ]

    tasks = []
    for d_sub in sorted(data_subfolders):
        label_subfolder = os.path.join(label_path, os.path.basename(d_sub))
        data_files = [
            f for f in os.listdir(d_sub)
            if f.lower().endswith('.nii.gz')]
        for d_file in data_files:
            base_name = d_file.replace('.nii.gz', '')
            label_file_path = os.path.join(label_subfolder, base_name+'_seg.nii.gz')
            if not os.path.exists(label_file_path):
                print(f"Label file not found for {label_file_path}, skipping...")
                continue
            data_file_path = os.path.join(d_sub, d_file)

            tasks.append((data_file_path, label_file_path, base_name,
                          output_image_path, output_label_path))

    if use_mp and len(tasks) > 0:
        from multiprocessing import Pool
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(process_item, tasks), total=len(tasks)):
                pass
    else:
        for task in tqdm(tasks):
            process_item(task)


def main():
    parser = argparse.ArgumentParser(description='Convert CTSpine1K nii.gz to mha')
    parser.add_argument('input_dir', type=str, help='输入目录，包含data和label子文件夹')
    parser.add_argument('out_dir', type=str, help='输出目录，将放置转换后的image和label文件夹')
    parser.add_argument('--mp', action='store_true', help='是否使用多进程')
    args = parser.parse_args()

    convert_ctspine_data(args.input_dir, args.out_dir, use_mp=args.mp)



if __name__ == '__main__':
    main()