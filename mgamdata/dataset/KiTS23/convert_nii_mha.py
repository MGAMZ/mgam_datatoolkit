import os
from mgamdata.dataset.base_convert import StandardFileFormatter

class KiTS23_formatter(StandardFileFormatter):
    @staticmethod
    def _series_id(image_path: str, label_path: str) -> str:
        ...
    
    def tasks(self) -> list:
        task_list = []
        for series_name in os.listdir(self.args.data_root):
            image_path = os.path.join(self.args.data_root, series_name, "imaging.nii.gz")
            label_path = os.path.join(self.args.data_root, series_name, "segmentation.nii.gz")
            task_list.append(
                (
                    image_path,
                    label_path,
                    self.args.dest_root,
                    series_name,
                    self.args.spacing,
                    self.args.size,
                )
            )
        return task_list

if __name__ == "__main__":
    formatter = KiTS23_formatter()
    formatter.execute()
