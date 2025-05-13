import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json


def extract_patches(image: sitk.Image,
                    label: sitk.Image | None,
                    patch_size: int | list[int],
                    patch_stride: int | list[int],
                    minimum_foreground_ratio: float = 0.0,
                    still_save_when_no_label: bool = False
                   ) -> list[tuple[sitk.Image, sitk.Image | None]]:
    """
    Extract 3D patches from image and label using a sliding window.
    Only patches with label foreground ratio >= minimum_foreground_ratio
    are kept. If still_save_when_no_label is True, patches without any
    foreground are also saved.
    Returns a list of tuples (image_patch, label_patch).
    """
    # determine no-label mode and load image array
    no_label = (label is None)
    img_arr = sitk.GetArrayFromImage(image)
    if not no_label:
        lbl_arr = sitk.GetArrayFromImage(label)
    
    # normalize patch_size and stride to 3-tuple (Z, Y, X)
    def to_triplet(x):
        if isinstance(x, int):
            return (x, x, x)
        if isinstance(x, (list, tuple)) and len(x) == 3:
            return tuple(x)
        raise ValueError('patch_size and patch_stride must be int or 3-length list/tuple')
    pZ, pY, pX = to_triplet(patch_size)
    sZ, sY, sX = to_triplet(patch_stride)
    Z, Y, X = img_arr.shape
    # drop sample if patch size larger than volume
    if pZ > Z or pY > Y or pX > X:
        return []
    
    # get spatial metadata for positioning patches
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    
    # compute start indices to cover full extent without padding
    def compute_starts(L, p, s):
        starts = list(range(0, L - p + 1, s))
        if starts[-1] != L - p:
            starts.append(L - p)
        return starts
    z_starts = compute_starts(Z, pZ, sZ)
    y_starts = compute_starts(Y, pY, sY)
    x_starts = compute_starts(X, pX, sX)
    
    patches = []
    # sliding window over computed starts
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                img_patch_np = img_arr[z:z+pZ, y:y+pY, x:x+pX]
                # decide saving
                if no_label:
                    if not still_save_when_no_label:
                        continue
                else:
                    lbl_patch_np = lbl_arr[z:z+pZ, y:y+pY, x:x+pX]
                    fg_count = int((lbl_patch_np > 0).sum())
                    fg_ratio = fg_count / (pZ * pY * pX)
                    if not (fg_ratio >= minimum_foreground_ratio or (fg_count == 0 and still_save_when_no_label)):
                        continue
                # convert back to sitk.Image
                img_patch = sitk.GetImageFromArray(img_patch_np)
                # set metadata for image
                new_origin = (
                    origin[0] + x * spacing[0],
                    origin[1] + y * spacing[1],
                    origin[2] + z * spacing[2]
                )
                img_patch.SetOrigin(new_origin)
                img_patch.SetSpacing(spacing)
                img_patch.SetDirection(direction)
                # convert and set label if exists
                if no_label:
                    lbl_patch = None
                else:
                    lbl_patch = sitk.GetImageFromArray(lbl_patch_np)
                    lbl_patch.SetOrigin(new_origin)
                    lbl_patch.SetSpacing(spacing)
                    lbl_patch.SetDirection(direction)
                patches.append((img_patch, lbl_patch))
    return patches

# --- Command-line interface for patch extraction tasks ---

def parse_args():
    parser = argparse.ArgumentParser(description="Extract patches from a folder of MHA images")
    parser.add_argument('src_folder', type=Path,
                        help='Folder containing `image` and `label` subfolders')
    parser.add_argument('dst_folder', type=Path,
                        help='Destination root folder to save patches')
    parser.add_argument('--patch-size', type=int, nargs='+', required=True,
                        help='Patch size as int or three ints (Z Y X)')
    parser.add_argument('--patch-stride', type=int, nargs='+', required=True,
                        help='Patch stride as int or three ints (Z Y X)')
    parser.add_argument('--minimum-foreground-ratio', type=float, default=0.0,
                        help='Minimum label foreground ratio to keep patch')
    parser.add_argument('--still-save-when-no-label', action='store_true',
                        help='If label missing, still extract patches unconditionally')
    parser.add_argument('--mp', action='store_true',
                        help='Use multiprocessing to process cases')
    return parser.parse_args()


def find_pairs(src_folder: Path):
    image_dir = src_folder / 'image'
    label_dir = src_folder / 'label'
    pairs = []
    for img_path in sorted(image_dir.glob('*.mha')):
        lbl_path = label_dir / img_path.name
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs


def process_case(args):
    img_path, lbl_path, dst_folder, patch_size, patch_stride, min_fg, still_save = args
    case_name = img_path.stem
    try:
        # create one folder per case
        out_case = dst_folder / case_name
        out_case.mkdir(parents=True, exist_ok=True)
        image = sitk.ReadImage(str(img_path))
        label = sitk.ReadImage(str(lbl_path))
        # load image array for metadata
        img_arr = sitk.GetArrayFromImage(image)
        shape = list(img_arr.shape)
        class_within_patch = {}
        patches = extract_patches(image, label,
                                  patch_size, patch_stride,
                                  min_fg, still_save)
        for idx, (img_patch, lbl_patch) in enumerate(patches):
            # write image and label patches with explicit suffix
            fname_img = f"{case_name}_{idx}_image.mha"
            sitk.WriteImage(img_patch, str(out_case / fname_img), True)
            if lbl_patch is not None:
                fname_lbl = f"{case_name}_{idx}_label.mha"
                # compute unique classes in this patch
                lbl_np = sitk.GetArrayFromImage(lbl_patch)
                class_within_patch[fname_lbl] = np.unique(lbl_np).tolist()
                sitk.WriteImage(lbl_patch, str(out_case / fname_lbl), True)
        # write series metadata
        series_meta = {
            "series_id": case_name,
            "shape": shape,
            "num_patches": len(patches),
            "anno_available": True,
            "class_within_patch": class_within_patch
        }
        with open(out_case / "SeriesMeta.json", "w") as f:
            json.dump(series_meta, f, indent=4)
        return case_name, len(patches)
    except Exception as e:
        print(f"Failed processing case {case_name}: {e}")
        return None


def main():
    args = parse_args()
    tasks = find_pairs(args.src_folder)
    task_args = [(
        img, lbl,
        args.dst_folder,
        args.patch_size,
        args.patch_stride,
        args.minimum_foreground_ratio,
        args.still_save_when_no_label
    ) for img, lbl in tasks]
    results = []
    if args.mp:
        with Pool(cpu_count()) as pool, tqdm(total=len(task_args), desc="Processing cases (mp)") as pbar:
            for res in pool.imap_unordered(process_case, task_args):
                results.append(res)
                pbar.update(1)
    else:
        with tqdm(total=len(task_args), desc="Processing cases") as pbar:
            for t in task_args:
                results.append(process_case(t))
                pbar.update(1)
    # removed summary prints
    # filter out failed cases
    valid_results = [r for r in results if r is not None]
    # write overall crop metadata
    crop_meta = {
        "src_folder": str(args.src_folder),
        "dst_folder": str(args.dst_folder),
        "patch_size": args.patch_size,
        "patch_stride": args.patch_stride,
        "anno_available": [case for case, count in valid_results]
    }
    with open(args.dst_folder / "crop_meta.json", "w") as f:
        json.dump(crop_meta, f, indent=4)


if __name__ == '__main__':
    main()
