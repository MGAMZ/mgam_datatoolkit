import os
import random
import re
from collections.abc import Generator

from ..dataset import mgam_BaseSegDataset
from .meta import CLASS_INDEX_MAP, DATA_ROOT_3D_MHA


class ImageTBAD_Seg3DDataset(mgam_BaseSegDataset):
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))
    SPLIT_RATIO = [0.7, 0.15, 0.15]

    def _split(self):
        all_series = [file.replace('.mha', '') 
                      for file in os.listdir(os.path.join(DATA_ROOT_3D_MHA, 'label'))
                      if file.endswith('.mha')]
        all_series = sorted(all_series, key=lambda x: abs(int(re.search(r'\d+', x).group())))
        random.shuffle(all_series)
        total = len(all_series)
        train_end = int(total * self.SPLIT_RATIO[0])
        val_end = train_end + int(total * self.SPLIT_RATIO[1])
        
        if self.split == 'train':
            return all_series[:train_end]
        elif self.split == 'val':
            return all_series[train_end:val_end]
        elif self.split == 'test':
            return all_series[val_end:]
        else:
            raise RuntimeError(f"Unsupported split: {self.split}")

    def sample_iterator(self) -> Generator[tuple[str, str], None, None]:
        for series in self._split():
            yield (os.path.join(self.data_root, 'image', series+'.mha'),
                   os.path.join(self.data_root, 'label', series+'.mha'))


class TBAD_3Dnpz(ImageTBAD_Seg3DDataset):
    def iter_series(self):
        for series in self._split():
            series_folder = os.path.join(self.data_root, series)
            for sample in os.listdir(series_folder):
                yield(os.path.join(series_folder, sample),
                      os.path.join(series_folder, sample))
