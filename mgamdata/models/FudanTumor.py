import pdb

import torch

from mmpretrain.datasets.transforms.formatting import PackInputs, DataSample
from mmpretrain.models.backbones.base_backbone import BaseBackbone

class PackTwoFeats(PackInputs):
    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        data_sample = DataSample()
        data_sample.set_gt_label(results['gt_label'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results


class MLP(BaseBackbone):
    def __init__(self, in_channels: int, hidden_channels:list[int],  *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.layers = self._build_layers()
    
    def _build_layers(self):
        layers = []
        in_channels = self.in_channels
        for hidden_channels in self.hidden_channels:
            layers.append(torch.nn.Linear(in_channels, hidden_channels))
            layers.append(torch.nn.ReLU())
            in_channels = hidden_channels
        return torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x),
