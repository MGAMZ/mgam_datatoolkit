import os
import json
import glob
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool



def compute_stats(arr):
    # 增加总和和元素个数用于计算全局统计量
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "shape": list(arr.shape),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "sum": float(np.sum(arr)),
        "n": int(arr.size),
        "sum_sq": float(np.sum(arr ** 2))
    }


def update_global_stats(global_stats, stats):
    if not global_stats:
        global_stats.update({
            "min": stats["min"],
            "max": stats["max"],
            "sum": stats["sum"],
            "n": stats["n"],
            "sum_sq": stats["sum_sq"]
        })
    else:
        global_stats["min"] = min(global_stats["min"], stats["min"])
        global_stats["max"] = max(global_stats["max"], stats["max"])
        global_stats["sum"] += stats["sum"]
        global_stats["n"] += stats["n"]
        global_stats["sum_sq"] += stats["sum_sq"]
    return global_stats


def finalize_global_stats(global_stats):
    mean = global_stats["sum"] / global_stats["n"]
    variance = (global_stats["sum_sq"] / global_stats["n"]) - (mean ** 2)
    return {
        "min": float(global_stats["min"]),
        "max": float(global_stats["max"]),
        "mean": float(mean),
        "std": float(np.sqrt(variance))
    }


def process_single_file(filepath):
    sitk_img = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(sitk_img)
    stats = compute_stats(arr)
    return os.path.basename(filepath), stats


def main(data_root, use_mp=False, num_workers=None):
    image_path = os.path.join(data_root, "image")
    mha_files = glob.glob(os.path.join(image_path, "*.mha"))
    
    per_file_stats = {}
    global_stats = {}
    
    if use_mp:
        with Pool(processes=num_workers) as pool:
            for filename, stats in tqdm(
                pool.imap_unordered(process_single_file, mha_files),
                total=len(mha_files),
                desc="Computing statistics (MP)",
                dynamic_ncols=True,
                leave=False
            ):
                per_file_stats[filename] = stats
                global_stats = update_global_stats(global_stats, stats)
    else:
        for f in tqdm(mha_files, 
                     desc="Computing statistics",
                     dynamic_ncols=True,
                     leave=False
        ):
            filename, stats = process_single_file(f)
            per_file_stats[filename] = stats
            global_stats = update_global_stats(global_stats, stats)

    global_stats = finalize_global_stats(global_stats)

    with open(os.path.join(data_root, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump(per_file_stats, f, ensure_ascii=False, indent=4)

    with open(os.path.join(data_root, "global_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("--mp", action="store_true", help="Enable multiprocessing")
    parser.add_argument("--workers", type=int, default=None, 
                       help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()
    
    main(args.data_root, args.mp, args.workers)