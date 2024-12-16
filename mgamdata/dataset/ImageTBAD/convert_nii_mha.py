import os
from ..base_convert import StandardFileFormatter

class ImageTBAD_formatter(StandardFileFormatter):
    @staticmethod
    def _series_id(image_path: str, label_path: str) -> str:
        return os.path.basename(image_path).replace("_image.nii.gz", "")
    
    def tasks(self) -> list:
        task_list = []
        for series_name in os.listdir(self.data_root):
            if series_name.endswith(".nii.gz"):
                image_path = os.path.join(self.data_root, series_name)
                label_path = image_path.replace("image", "label")
                series_id = self._series_id(image_path, label_path)
                task_list.append(
                    (
                        image_path,
                        label_path,
                        self.dest_root,
                        series_id,
                        self.spacing,
                        self.size,
                    )
                )
        return task_list

if __name__ == "__main__":
    formatter = ImageTBAD_formatter.start_from_argparse()
    formatter.execute()
