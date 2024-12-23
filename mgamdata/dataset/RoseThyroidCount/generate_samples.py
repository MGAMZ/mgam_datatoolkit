import os
import argparse
import pdb
import multiprocessing as mp
import json
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd


def __reference_implementation__():
    import json
    from glob import glob
    import cv2
    import numpy as np

    cell_path = ...
    mask_path = "/fileser/rongkuan/mywork/cellcount/QualityTROSE_cell_count/patch_mask/"
    f = glob("/home/rongkuan/Quality_TROSE/7787/*.json")
    cell_count = 0
    cell_count2 = 0
    alpha = 1
    beta = 0.2
    gamma = 0
    for label in f:
        with open(label, "r") as file:
            str = file.read()
            label_name = label.split("/")[-1].split(".")[0]

            ori_path = matchingdf[matchingdf["序列号"] == label_name]["原始路径"].iloc[0] # type:ignore
            real_uid = ori_path.split("/")[0].split("_")[0]
            real_filename = ori_path.split("/")[-1]
            real_filename = real_uid + "_" + real_filename

            train_img = glob(cell_path + real_filename)[0]

            maskpath = mask_path + real_filename
            maskpath = maskpath.replace(".png", ".npy")
            print(maskpath)
            ori_img = cv2.imread(train_img)
            data = json.loads(str)
            mask = np.zeros((64, 64))
            cell_count = 0
            cell_count2 = 0
            for i in data["shapes"]:
                y = int(i["points"][0][0] / 4 / 2)
                x = int(i["points"][0][1] / 4 / 2)

                y = min(y, 63)
                x = min(x, 63)
                mask[x][y] += 1

            mask1 = cv2.GaussianBlur(mask, (7, 7), 1.5)
            mask2 = cv2.GaussianBlur(mask, (11, 11), 2)

            cell_count = mask.sum()
            cell_count2 = mask2.sum()
            print(cell_count, cell_count2, len(data["shapes"]))

            np.save(maskpath, mask2)


class FileID_Map:
    def __init__(self, file_id_map_path: str):
        self.csv = pd.read_csv(file_id_map_path)

    def search_from_file_path(self, file_path: str):
        found_patch = self.csv[self.csv["originPath"] == file_path]
        if found_patch.empty:
            # tqdm.write(f"File Path `{file_path}` not found in FileID_Map")
            return None
        else:
            return found_patch["seriesinstanceUID"].values[0]


class AnnoReader:
    def __init__(self, location_anno: str, cluster_anno: str):
        self.location_anno = pd.read_csv(location_anno)
        self.cluster_anno = pd.read_csv(cluster_anno)

    def search_from_file_path(self, SeriesID: str):
        found_location_labels = self.location_anno[
            self.location_anno["序列编号"] == SeriesID
        ]["影像结果"]
        found_cluster_labels = self.cluster_anno[
            self.cluster_anno["序列编号"] == SeriesID
        ]["是否成团"]

        if found_location_labels.empty:
            # tqdm.write(f"Series ID `{SeriesID}` not found in Annotation")
            found_location_labels = None
        else:
            found_location_labels = found_location_labels.values
        if found_cluster_labels.empty:
            found_cluster_labels = None
        else:
            found_cluster_labels = found_cluster_labels.values

        return found_location_labels, found_cluster_labels


def convert_one_patch_to_heatmap(
    patch_path: str, location_anno, cluster_anno, save_path: str
):
    assert save_path.endswith(".npz"), f"Save path must be a PNG file, got: {save_path}"

    patch_img = cv2.imread(patch_path)
    patch_x, patch_y = patch_img.shape[:2]
    heatmap = np.zeros((patch_x, patch_y))

    # anno format: ['{"point1":{"x":192.92,"y":154.07,"z":1}}',
    #               '{"point1":{"x":192.92,"y":154.07,"z":1}}',
    #              ...]
    for point in location_anno:
        point_info = json.loads(point)["point1"]
        x = np.round(point_info["x"]).astype(np.int32)
        y = np.round(point_info["y"]).astype(np.int32)
        if x >= patch_x or y >= patch_y:
            tqdm.write(f"Point ({x}, {y}) out of bound")
            continue
        heatmap[y, x] += 1

    np.savez_compressed(
        save_path, img=patch_img, gt_seg_map=heatmap, gt_cluster=cluster_anno
    )
    tqdm.write(f"Saved Heatmap to {save_path}")

    return heatmap, heatmap.sum(), save_path


def convert_gt_points_to_heatmap(
    img_root: str,
    location_anno_path: str,
    cluster_anno_path: str,
    file_id_map_path: str,
    out_root: str | None,
    use_mp: bool,
):
    mapper = FileID_Map(file_id_map_path)
    anno_reader = AnnoReader(location_anno_path, cluster_anno_path)
    count_instance = 0
    count_valid_instance = 0
    count_seriesID_NotFound = 0
    count_location_anno_NotFound = 0
    count_cluster_anno_NotFound = 0
    count_anno = 0
    if out_root is not None:
        if use_mp:
            p = mp.Pool(1)
        p_results = []
        os.makedirs(out_root, exist_ok=True)

    # allocate tasks
    for instance in tqdm(
        os.listdir(img_root),
        desc="Generating labels",
        dynamic_ncols=True,
        leave=True,
    ):
        instance_folder = os.path.join(img_root, instance)

        for patch in os.listdir(instance_folder):
            count_instance += 1
            
            patch_ident_path = os.path.join(instance, patch)
            patch_full_path = os.path.join(instance_folder, patch)

            series_id = mapper.search_from_file_path(patch_ident_path)
            if series_id is None:
                count_seriesID_NotFound += 1
                continue

            location_anno, cluster_anno = anno_reader.search_from_file_path(series_id)
            if location_anno is None:
                count_location_anno_NotFound += 1
                continue
            if cluster_anno is None:
                count_cluster_anno_NotFound += 1
                continue
            
            count_valid_instance += 1
            
            if out_root is not None:
                save_path = os.path.join(
                    out_root, patch_ident_path.replace(".png", ".npz")
                )
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                if use_mp:
                    result = p.apply_async(
                        convert_one_patch_to_heatmap,
                        args=(patch_full_path, location_anno, cluster_anno, save_path),
                    )
                    p_results.append(result)
                else:
                    result = convert_one_patch_to_heatmap(
                        patch_full_path, location_anno, cluster_anno, save_path
                    )
                    p_results.append(result)

    # fetch results
    if out_root is not None:
        if use_mp:
            p.close()
            p.join()
            for result in p_results:
                heatmap, cell_count, save_path = result.get()
                count_anno += cell_count
                tqdm.write(f"Saved Heatmap to {save_path}")
        else:
            for heatmap, cell_count, save_path in p_results:
                count_anno += cell_count
                tqdm.write(f"Saved Heatmap to {save_path}")

    print(f"Total Instance: {count_instance}")
    print(f"SeriesID Not Found: {count_seriesID_NotFound}")
    print(f"Location Annotation Not Found: {count_location_anno_NotFound}")
    print(f"Cluster Annotation Not Found: {count_cluster_anno_NotFound}")
    print(f"Valid Instance: {count_valid_instance}")
    print(f"Valid Cells: {count_anno}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert GT Points to Heatmap")
    parser.add_argument(
        "--patch-root", type=str, help="Root directory of image data"
    )
    parser.add_argument(
        "--location-anno-csv",
        type=str,
        help="CSV file containing location annotations",
    )
    parser.add_argument(
        "--cluster-anno-csv",
        type=str,
        help="CSV file containing cluster annotations",
    )
    parser.add_argument(
        "--fileidmap-csv", 
        type=str, 
        help="CSV file containing file ID mapping",
    )
    parser.add_argument(
        "--out-root", type=str, default=None, help="Root directory to save heatmaps"
    )
    parser.add_argument("--mp", action="store_true", help="Use multiprocessing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_gt_points_to_heatmap(
        args.patch_root,
        args.location_anno_csv,
        args.cluster_anno_csv,
        args.fileidmap_csv,
        args.out_root,
        args.mp,
    )
