import os
import pdb
import re 

import numpy as np
from ..base import mgam_BaseSegDataset



class mgam_Standard_2D_png(mgam_BaseSegDataset):
    def __init__(self, *args, **kwargs) -> None:
        # HACK: Most implementations use the more elastic dataset,
        # which is `mgam_SemiSup_2D_png`, and it contains a `mode` parameter.
        kwargs.pop("mode", None)
        super().__init__(*args, **kwargs)
        self.data_root: str

    def _split(self):
        all_series = [
            file.replace(".png", "")
            for file in os.listdir(os.path.join(self.data_root, "label"))
            if file.endswith(".png")
        ]
        all_series = sorted(
            all_series, key=lambda x: abs(int(re.search(r"\d+", x).group()))
        )
        np.random.shuffle(all_series)
        total = len(all_series)
        train_end = int(total * self.SPLIT_RATIO[0])
        val_end = train_end + int(total * self.SPLIT_RATIO[1])

        if self.split == "train":
            return all_series[:train_end]
        elif self.split == "val":
            return all_series[train_end:val_end]
        elif self.split == "test":
            return all_series[val_end:]
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")

    def sample_iterator(self):
        for series in self._split():
            image_png_path = os.path.join(self.data_root, "image", series + ".png")
            label_png_path = os.path.join(self.data_root, "label", series + ".png")
            if os.path.exists(image_png_path) and os.path.exists(label_png_path):
                yield (image_png_path, label_png_path)