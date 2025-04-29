import os
from mgamdata.dataset.base_convert import StandardFileFormatter

class FLARE2022_formatter(StandardFileFormatter):
    @property
    def _unsup_folders(self):
        return [
            os.path.join(self.args.data_root, "Training", "FLARE_UnlabeledCase1-1000"),
            os.path.join(self.args.data_root, "Training", "FLARE_UnlabeledCase1001-2000"),
        ]
    
    @property
    def _sup_folder(self):
        return os.path.join(self.args.data_root, "Training", "FLARE_LabeledCase50")
    
    @staticmethod
    def _series_id(image_path, label_path:str|None):
        return os.path.basename(image_path).replace("_0000.nii.gz", "").replace("Case_", "")
    
    def tasks(self) -> list:
        task_list = []
        
        for unsup_folders in self._unsup_folders:
            for series_name in os.listdir(unsup_folders):
                if series_name.endswith(".nii.gz"):
                    image_path = os.path.join(unsup_folders, series_name)
                    series_id = self._series_id(image_path, None)
                    task_list.append(
                        (
                            image_path,
                            None,
                            self.args.dest_root,
                            series_id,
                            self.args.spacing,
                            self.args.size,
                        )
                    )
        
        sup_img_folder = os.path.join(self._sup_folder, "images")
        sup_ann_folder = os.path.join(self._sup_folder, "labels")
        for series_name in os.listdir(sup_img_folder):
            if series_name.endswith(".nii.gz"):
                image_path = os.path.join(sup_img_folder, series_name)
                label_path = os.path.join(sup_ann_folder, series_name.replace("_0000.nii.gz", ".nii.gz"))
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
        
        return task_list

if __name__ == "__main__":
    formatter = FLARE2022_formatter()
    formatter.execute()
