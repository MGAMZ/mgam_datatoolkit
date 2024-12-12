import pdb

import torch
from torch import nn

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


# 第一版模型，用于跑通程序以及简单观察可拟合性
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


class YuTing_RFSS(BaseBackbone):
    def __init__(self, in_channels: int, hidden_channels:int=448, out_channels:int=22, *args, **kwargs):
        super(YuTing_RFSS, self).__init__(*args, **kwargs)
        self.hidden1 = nn.Linear(in_features=in_channels, out_features=hidden_channels, bias=True)
        self.hidden2 = nn.Linear(hidden_channels, hidden_channels//2)
        self.hidden3 = nn.Linear(hidden_channels//2, hidden_channels//4)
        self.hidden4 = nn.Linear(hidden_channels//4, hidden_channels//8)
        self.rfss_predict = nn.Linear(hidden_channels//8, out_channels)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.activation(self.hidden4(x))
        return x,


class YuTing_RFSS_svp(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(YuTing_RFSS_svp, self).__init__()
        self.hidden1 = nn.Linear(in_features=in_channels, out_features=hidden_channels, bias=True)
        self.hidden2 = nn.Linear(hidden_channels, hidden_channels//2)
        self.hidden3 = nn.Linear(hidden_channels//2, hidden_channels//4)
        self.hidden_cat = nn.Linear(hidden_channels // 4 + 2, hidden_channels // 8)
        self.activation = nn.ReLU()
        
    def forward(self, x, svp):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x_cat = torch.cat((x, svp), dim = 1)
        x = self.activation(self.hidden_cat(x_cat))
        return x,


class YiQin_WeightedPatch(BaseBackbone):
    def __init__(self, num_CLAM_feats:int, num_heads:int, out_CLAM_feat_channels:int=64, *args, **kwargs):
        super(YiQin_WeightedPatch, self).__init__(*args, **kwargs)
        self.num_CLAM_feats = num_CLAM_feats
        self.num_heads = num_heads
        self.out_CLAM_feat_channels = out_CLAM_feat_channels
        self.extractor = nn.Sequential(
            nn.MultiheadAttention(embed_dim=num_CLAM_feats, num_heads=num_heads, batch_first=True), 
            nn.Linear(1024, 256), 
            nn.MultiheadAttention(embed_dim=num_CLAM_feats, num_heads=num_heads, batch_first=True), 
            nn.Linear(256, 64), 
            nn.MultiheadAttention(embed_dim=num_CLAM_feats, num_heads=num_heads, batch_first=True), 
            nn.Linear(64, 1)
        )
        self.channel_compress = nn.Linear(num_CLAM_feats, out_CLAM_feat_channels)
    
    def forward(self, CLAM_feat: torch.Tensor, CSV_feat: torch.Tensor):
        """
        Args:
            CLAM_feat (torch.Tensor): [N, C, 1024]
            csv_feat  (torch.Tensor): [N, 69]
        Returns:
            torch.Tensor: [N, out_CLAM_feat_channels + 69]
        """
        CLAM_compress_weight = self.extractor(CLAM_feat).squeeze() # [N, C, 1024] -> [N, C]
        patchwise_compressed = torch.matmul(CLAM_compress_weight, CLAM_feat) # [N, C] * [N, C, 1024] -> [N, 1024]
        channelwise_compressed = self.channel_compress(patchwise_compressed) # [N, 1024] -> [N, out_CLAM_feat_channels]
        concat_feat = torch.cat((channelwise_compressed, CSV_feat), dim=1) # [N, out_CLAM_feat_channels + 64]
        return concat_feat,