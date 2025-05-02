import os
import pdb

from mgamdata.dataset.base_convert import StandardFileFormatter



class CTSpine1K_Formatter(StandardFileFormatter):
    SUBFOLDERS = ["colon", "COVID-19", "HNSCC-3DCT-RT_neck", "liver"]
    
    def tasks(self) -> list:
        task_list = []

        data_folder = os.path.join(self.args.data_root, "data")
        label_folder = os.path.join(self.args.data_root, "label")

        for subfolder in self.SUBFOLDERS:
            subfolder_data_path = os.path.join(data_folder, subfolder)
            subfolder_label_path = os.path.join(label_folder, subfolder)
            if not os.path.exists(subfolder_data_path) or not os.path.exists(subfolder_label_path):
                print(f"Data or label folder {subfolder_data_path} not found, skipping")
                continue
                
            for file_name in os.listdir(subfolder_data_path):
                if file_name.endswith(".nii.gz"):
                    image_path = os.path.join(subfolder_data_path, file_name)
                    label_path = os.path.join(subfolder_label_path, file_name.replace(".nii.gz", "_seg.nii.gz"))
                    
                    if not os.path.exists(label_path):
                        print(f"Label file {label_path} not found, skipping")
                        continue
                    series_id = self._series_id(image_path, label_path)

                    task_list.append(
                        (
                            image_path,
                            label_path,
                            self.args.dest_root,
                            series_id,
                            self.args.spacing,
                            self.args.size,
                        )
                    )
        
        print(f"Found {len(task_list)} matching image-label pairs in CTSpine1K dataset")
        return task_list


if __name__ == "__main__":
    formatter = CTSpine1K_Formatter()
    formatter.execute()