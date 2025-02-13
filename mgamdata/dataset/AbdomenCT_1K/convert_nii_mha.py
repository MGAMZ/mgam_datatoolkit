import os
from mgamdata.dataset.base_convert import format_from_nnUNet

class AbdomenCT1K_formatter(format_from_nnUNet):
    @staticmethod
    def _series_id(image_path: str, label_path: str) -> str:
        return os.path.basename(label_path).replace(".nii.gz", "").replace("Case_", "")
    
if __name__ == "__main__":
    formatter = AbdomenCT1K_formatter()
    formatter.execute()
