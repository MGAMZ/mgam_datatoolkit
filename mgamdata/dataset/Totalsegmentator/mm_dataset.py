import os
from os.path import join
from tqdm import tqdm

import orjson
import pandas as pd

from ..base import mgam_BaseSegDataset
from .meta import CLASS_INDEX_MAP, get_subset_and_rectify_map


class TotalsegmentatorIndexer:

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.index_file = join(self.data_root, f'index.json')

        if not os.path.exists(self.index_file):
            self.generate_index_json_file()
        with open(self.index_file, 'rb') as f:
            self.img_index = orjson.loads(f.read())

    def generate_index_json_file(self):
        index = {
            split: list(self._index(join(self.data_root, 'img_dir'), split))
            for split in ['train', 'val', 'test']
        }
        with open(self.index_file, 'wb') as f:
            f.write(orjson.dumps(index, option=orjson.OPT_INDENT_2))

    def _index(self, image_root: str, split: str):
        split_folder = join(image_root, split)
        for series in tqdm(iterable=os.listdir(split_folder),
                           desc=f"Indexing {split} split",
                           dynamic_ncols=True,
                           leave=False):
            series_folder = join(split_folder, series)
            image_paths = sorted(os.listdir(series_folder))
            for image_path in image_paths:
                image_path = join(series_folder, image_path)
                yield os.path.relpath(image_path, self.data_root)

    def fetcher(self, split: str):
        selected_split_image_paths: list = self.img_index[split]
        return [(os.path.join(self.data_root, image_path),
                 os.path.join(self.data_root,
                              image_path.replace('img_dir', 'ann_dir')))
                for image_path in selected_split_image_paths]


class Tsd_base(mgam_BaseSegDataset):
    METAINFO = dict(classes=list(CLASS_INDEX_MAP.keys()))

    def __init__(self, meta_csv:str, subset: str|None=None, **kwargs) -> None:
        self.meta_table = pd.read_csv(meta_csv)

        if subset is not None and subset != 'all':
            new_classes = list(get_subset_and_rectify_map(subset)[0].keys())
        else:
            new_classes = self.METAINFO['classes']

        super().__init__(metainfo={'classes': new_classes}, **kwargs)

    def _split(self):
        activate_series = self.meta_table[self.meta_table['split']==self.split]
        return activate_series['image_id'].tolist()


class Tsd_Mha(Tsd_base):
    def sample_iterator(self):
        for series in self._split():
            img_mha_path = os.path.join(self.data_root, 'image', f'{series}.mha')
            lbl_mha_path = os.path.join(self.data_root, 'label', f'{series}.mha')
            if os.path.exists(img_mha_path) and os.path.exists(lbl_mha_path):
                yield (img_mha_path, lbl_mha_path)


class Tsd3D_PreCrop_Npz(Tsd_Mha):
    def sample_iterator(self):
        for series in self._split():
            samples = os.path.join(self.data_root, series)
            if os.path.exists(samples):
                for cropped_sample in os.listdir(samples):
                    if cropped_sample.endswith('.npz'):
                        yield (os.path.join(samples, cropped_sample),
                            os.path.join(samples, cropped_sample))



