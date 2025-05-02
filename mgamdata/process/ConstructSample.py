import os
import pdb
import json
from tqdm import tqdm
from typing_extensions import Mapping, Sequence

import numpy as np
from torch import Tensor

from mmengine.config import Config
from mmengine.structures import BaseDataElement
from mmengine.dataset import DefaultSampler

from mgamdata.mm.mmeng_PlugIn import DynamicRunnerSelection


class ConstructSample:
    def __init__(
        self,
        sample_root: str,
        cfg_path:str, 
        *args,
        **kwargs,
    ):
        """
        Args:=
            sample_root (str): root path of sample
            cfg_path (str): config files of OpenMM
        """
        self.sample_root = sample_root
        self.cfg = Config.fromfile(cfg_path)
        self._override_cfg()
        self.runner = DynamicRunnerSelection(self.cfg)

    def _override_cfg(self):
        self.cfg.work_dir = self.sample_root
        for name in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            if hasattr(self, name):
                self.cfg.get(name)["sampler"] = dict(
                    type=DefaultSampler, 
                    shuffle=False
                )
                self.cfg.get(name)["persistent_workers"] = False

    def _log(self) -> dict:
        """Log the construction configurations."""
        return dict()

    def cast_data(self, data: dict) -> dict:
        """
        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at numpy format.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, "_fields"):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)
        elif isinstance(data, (Tensor, BaseDataElement)):
            return data.cpu().numpy()
        else:
            return data

    def construct(self, split:str):
        if split == "train":
            loader = self.runner._train_dataloader
        elif split == "val":
            loader = self.runner._val_dataloader
        elif split == "test":
            loader = self.runner._test_dataloader
        else:
            raise KeyError(f"Not Supported Split {split}.")
        
        loader = self.runner.build_dataloader(loader)
        
        msg = self._log()
        msg["num_samples_planed"] = len(loader)
        json.dump(msg, open(os.path.join(self.sample_root, "log.json"), "w"), indent=4)
        save_folder = os.path.join(self.sample_root, f"{split}Set")
        os.makedirs(save_folder, exist_ok=True)

        pbar = tqdm(
            loader,
            total=len(loader),
            desc="Constructing",
            dynamic_ncols=True,
            leave=False,
        )
        for i, sample in enumerate(pbar):
            sample = self.cast_data(sample)
            np.savez_compressed(
                os.path.join(save_folder, f"{i}.npz"), **sample
            )
        
        print(f"Finished constructing {len(loader)} samples.")


if __name__ == "__main__":
    constructer = ConstructSample(
        sample_root = "/file1/mgam_projects/FastSlow/ConstructedSamples",
        cfg_path = "/file1/mgam_projects/FastSlow/configs/0.1.0.9.LocalStart/MedNeXt.py"
    )
    constructer.construct("train")