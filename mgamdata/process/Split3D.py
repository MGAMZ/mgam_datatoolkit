import os
import argparse
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import multiprocessing



def load_mha(file_path: str) -> np.ndarray:
    itk = sitk.ReadImage(file_path)
    itk = sitk.DICOMOrient(itk, 'LPI')
    return sitk.GetArrayFromImage(itk)


def create_sliding_windows(
    volume: np.ndarray, 
    window_size: int, 
    stride: int
) -> list[tuple[int, np.ndarray]]:
    """
    在Z轴方向对3D体积进行滑动窗口采样。
    
    Args:
        volume: 3D体积数据，shape=[Z, Y, X]
        window_size: Z轴方向窗口大小
        stride: 滑动步长
    
    Returns:
        (start_idx, window_data) 的列表
    """
    z_length = volume.shape[0]
    windows = []
    
    # 常规窗口
    for start_idx in range(0, z_length - window_size + 1, stride):
        slice_data = volume[start_idx : start_idx + window_size]
        windows.append((start_idx, slice_data))
    
    # 处理可能的尾部不足窗口
    last_start_idx = stride * ((z_length - window_size) // stride)
    if last_start_idx + window_size < z_length:
        slice_data = volume[-window_size : ]
        windows.append((len(slice_data)-window_size, slice_data))
    
    return windows


def sample_volume(args):
    """
    对一个MHA文件(含图像与标签)进行Z轴滑动窗口采样，并在对应子目录下生成SeriesMeta.json文件。
    
    Args:
        image_path: 图像MHA文件路径
        label_path: 标签MHA文件路径
        output_dir: 输出目录
        window_size: Z轴方向窗口大小
        stride: 滑动步长
    
    Returns:
        (文件名, 采样数, 错误信息) 的元组
    """
    try:
        image_path, label_path, output_dir, window_size, stride, ensure_slice_foreground = args
        # 加载图像和标签
        image = load_mha(image_path)
        label = load_mha(label_path)
        if image.shape != label.shape:
            raise RuntimeError(f"图像与标签形状不匹配: image={image.shape}, label={label.shape}")
        if image.shape[0] < window_size:
            tqdm.write(f"{image_path} 的Z轴长度小于窗口大小，跳过处理。")
            return { 
                os.path.basename(image_path.replace('.mha', '')): {
                    "num_patches": 0,
                    "anno_available": False,
                    "cropped_center": None,
                }
            }
        
        # 创建输出文件夹
        series_id = os.path.splitext(os.path.basename(image_path))[0]
        series_folder = os.path.join(output_dir, series_id)
        
        # 生成滑动窗口
        image_windows = create_sliding_windows(image, window_size, stride)
        label_windows = create_sliding_windows(label, window_size, stride)
        
        # 用于记录JSON信息
        existed_classes: dict[str, list[int]] = {}
        cropped_center: list[tuple[float, float, float]] = []
        
        # 获取Y、X方向尺寸：注：volume.shape 为 [Z, Y, X]
        _, height, width = image.shape
        
        # 依次保存滑动窗口数据
        for idx, ((z_start, img_window), (_, label_window)) in enumerate(zip(image_windows, label_windows)):
            if ensure_slice_foreground is True and label_window.any(axis=(1,2)).all().item() is False:
                continue
            
            # 保存npz
            os.makedirs(series_folder, exist_ok=True)
            sample_name = f"{idx}.npz"
            save_path = os.path.join(series_folder, sample_name)
            np.savez_compressed(save_path, img=img_window, gt_seg_map=label_window)
            
            # 记录NPZ中的label类别
            unique_classes = np.unique(label_window).tolist()
            existed_classes[sample_name] = unique_classes
            
            # 计算窗口中心坐标(简单示例：只考虑Z方向实际范围，XY整幅)
            z_end = z_start + window_size
            z_center = (z_start + z_end) / 2
            y_center = height / 2
            x_center = width / 2
            cropped_center.append((z_center, y_center, x_center))
        
        num_patches = len(existed_classes)
        anno_available = (num_patches > 0)
        
        # 如果至少有一个patch, 获取窗口的形状(取第一个为准)
        patch_shape = image_windows[0][1].shape if num_patches > 0 else None
        
        # 生成JSON文件 "SeriesMeta.json"
        if anno_available is True:
            metadata_path = os.path.join(series_folder, "SeriesMeta.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "series_id": series_id,
                        "shape": patch_shape,
                        "num_patches": num_patches,
                        "anno_available": anno_available,
                        "class_within_patch": existed_classes,
                        "cropped_center": cropped_center,
                    },
                    f,
                    indent=4
                )
        
        return {
            os.path.basename(series_folder): {
                "num_patches": num_patches,
                "anno_available": anno_available,
                "cropped_center": cropped_center,
            }
        }
        
    except Exception as e:
        return os.path.basename(image_path), str(e)


def process_dataset(
    data_dir: str,
    output_dir: str,
    window_size: int,
    stride: int,
    use_mp: bool = False,
    num_workers: int|None = None,
    ensure_slice_foreground: bool = False,
) -> None:
    """
    对 data_dir 下的 image/ 和 label/ 目录进行遍历，分别执行滑动窗口采样。
    采样结果及其 JSON 文件会存到 output_dir 下对应的子文件夹内。
    """
    # 检查目录
    image_dir = os.path.join(data_dir, "image")
    label_dir = os.path.join(data_dir, "label")
    for d in [image_dir, label_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"目录不存在: {d}")
    
    # 找到所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".mha")]
    if not image_files:
        print(f"警告: 在 {image_dir} 中未找到任何MHA文件。")
        return
    
    print(f"在 {image_dir} 中找到了 {len(image_files)} 个MHA文件，开始处理...")
    
    # 组装处理参数
    tasks = []
    for img_file in image_files:
        image_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file)
        if not os.path.exists(label_path):
            print(f"警告: 与 {img_file} 对应的标签文件不存在: {label_path}")
            continue
        tasks.append((image_path, label_path, output_dir, window_size, stride, ensure_slice_foreground))
    
    # 处理文件
    results = {}
    if use_mp:
        num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        print(f"使用多进程处理, 进程数: {num_workers}")
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap_unordered(sample_volume, tasks),
                               total=len(tasks), 
                               desc="处理进度", 
                               dynamic_ncols=True):
                if isinstance(result, tuple):
                    series_id, error = result
                    print(f"处理 {series_id} 时发生错误: {error}")
                else:
                    results.update(result)
    else:
        for t in tqdm(tasks, desc="处理进度", dynamic_ncols=True):
            result = sample_volume(t)
            if isinstance(result, tuple):
                series_id, error = result
                print(f"处理 {series_id} 时发生错误: {error}")
            else:
                results.update(result)
    
    cropped_series_meta = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "window_size": window_size,
        "stride": stride,
        "num_series": len(results),
        "num_patches": sum([one_series_meta["num_patches"]
                            for one_series_meta in results.values()]),
        "anno_available": [series_id
                           for series_id, series_meta in results.items()
                           if series_meta["anno_available"] is True],
    }
    json.dump(cropped_series_meta, 
              open(os.path.join(output_dir, "crop_meta.json"), "w", encoding="utf-8"), indent=4)
    print(f"全部处理完成，采样结果元数据已保存到 {os.path.join(output_dir, 'crop_meta.json')}.")


def main():
    parser = argparse.ArgumentParser(description="对3D医学体积MHA文件进行Z轴滑动窗口采样，并生成JSON元数据。")
    parser.add_argument("data_root", help="数据根目录(包含image和label子目录)")
    parser.add_argument("output_dir", help="采样结果输出目录")
    parser.add_argument("--window-size", type=int, default=64, help="窗口大小(Z轴方向)")
    parser.add_argument("--stride", type=int, default=32, help="滑动步长")
    parser.add_argument("--mp", action="store_true", help="是否使用多进程处理")
    parser.add_argument("--num-workers", type=int, help="多进程时的进程数量，默认为CPU核心数-1")
    parser.add_argument("--ensure-slice-foreground", action="store_true", help="确保滑动窗口中至少包含一个前景像素")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        process_dataset(
            data_dir=args.data_root,
            output_dir=args.output_dir,
            window_size=args.window_size,
            stride=args.stride,
            use_mp=args.mp,
            num_workers=args.num_workers,
            ensure_slice_foreground=args.ensure_slice_foreground
        )
    except Exception as e:
        print(f"[致命错误] 运行过程中出现异常: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()