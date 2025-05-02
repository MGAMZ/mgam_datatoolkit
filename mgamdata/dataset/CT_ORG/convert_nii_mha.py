import os
from mgamdata.dataset.base_convert import StandardFileFormatter

class CT_ORG_formatter(StandardFileFormatter):
    @staticmethod
    def _series_id(image_path: str, label_path: str) -> str:
        return os.path.basename(image_path).replace("volume-", "").replace(".nii.gz", "")
    
    def tasks(self) -> list:
        task_list = []
        for series_name in os.listdir(self.args.data_root):
            if series_name.endswith(".nii.gz") and "volume" in series_name:
                image_path = os.path.join(self.args.data_root, series_name)
                label_path = image_path.replace("volume", "labels")
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
    formatter = CT_ORG_formatter()
    formatter.execute()
