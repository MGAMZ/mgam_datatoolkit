import os
import pdb
from tqdm import tqdm
from colorama import Style, Fore

import pandas as pd

from mgamdata.dataset.base_convert import StandardFileFormatter


class TCGA_MetaParser:
    META_SERIESUID_ATTR = "Series UID"
    META_PATH_ATTR = "File Location"

    def __init__(self, csv_file_path: str):
        self.df = pd.read_csv(csv_file_path)

    def get_series_info(self, series_uid:str):
        # Get all information for the given Series UID
        return self.df[self.df[self.META_SERIESUID_ATTR] == series_uid]
    
    def get_series_uids(self, attr:str, val) -> str:
        # Get all Series UID based on the given attribute and value
        
        result = self.df[self.df[attr] == val][self.META_SERIESUID_ATTR].values
        if len(result) == 0 or result[0] > 1:
            print(Fore.YELLOW, f"Series UID not found for {attr}={val}", Style.RESET_ALL)
        return result[0]

    def get_all_filtered_sample(self, attr:str, val) -> list[tuple[str, str]]:
        filtered_samples = self.df[self.df[attr] == val]
        seriesUIDs = filtered_samples[self.META_SERIESUID_ATTR]
        paths = filtered_samples[self.META_PATH_ATTR]
        return list(zip(seriesUIDs, paths))


class TCGA_Formatter(StandardFileFormatter):
    MINIMUM_DCM_SLICES = 16

    def execute(self):
        meta_path = os.path.join(self.args.data_root, "metadata.csv")
        assert os.path.exists(meta_path), (
            "A standard TCGA dataset downloaded using NBIA Data Retriever "
            "must have a metadata.csv in the root directory"
        )
        self.TCGA_meta = TCGA_MetaParser(meta_path)
        return super().execute()

    def _series_id(self, image_path: str, label_path: str | None) -> str:
        folder_rel_path = os.path.relpath(image_path, os.path.dirname(image_path)).replace("\\", "/")
        return self.TCGA_meta.get_series_uids(TCGA_MetaParser.META_PATH_ATTR, folder_rel_path)

    def tasks(self) -> list:
        task_list = []
        deprecated_dcm = 0
        samples = self.TCGA_meta.get_all_filtered_sample(attr="Modality", val="CT")
        for seriesUID, dcms_folder in tqdm(samples, desc="Searching"):
            dcms_folder = os.path.join(self.args.data_root, str(dcms_folder).replace("\\", "/"))
            if not os.path.exists(dcms_folder):
                print(Fore.YELLOW, f"Folder not found: {dcms_folder}", Style.RESET_ALL)
                continue
            
            dcm_files = [os.path.join(dcms_folder, f) 
                         for f in os.listdir(dcms_folder) 
                         if f.lower().endswith(".dcm")]

            # dcm files
            if len(dcm_files) >= self.MINIMUM_DCM_SLICES:
                tqdm.write(f"Found available dcm series: {seriesUID}")
                label_path = None
                task_list.append(
                    (
                        dcm_files[0], # Need only one sample, the loader will automatically determine all slices.
                        label_path,
                        self.args.dest_root,
                        seriesUID,
                        self.args.spacing,
                        self.args.size,
                    )
                )
            else:
                deprecated_dcm += 1

        print(
            f"Total {len(task_list)+deprecated_dcm} series, "
            f"among which {len(task_list)} available series, "
            f"{deprecated_dcm} deprecated dcms series."
        )
        return task_list


if __name__ == "__main__":
    formatter = TCGA_Formatter()
    formatter.execute()
