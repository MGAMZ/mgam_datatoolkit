import pdb

import torch
from torch import nn
from mmcv.transforms import BaseTransform

from mmengine.model.base_model import BaseDataPreprocessor
from mmpretrain.datasets.transforms.formatting import PackInputs, DataSample
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.models.classifiers.image import ImageClassifier


NUM_CSV_FEAT = 68
NUM_CLAM_FEAT_CHANNEL = 1024


class PathologyPreprocessor(BaseDataPreprocessor):
    def __init__(self, *args, **kwargs):
        super(PathologyPreprocessor, self).__init__()

    def forward(self, data: dict, training: bool = False) -> dict:
        inputs = torch.stack(self.cast_data(data["inputs"]))
        data_samples = data.get("data_samples", None)
        data_samples = self.cast_data(data_samples)
        return {"inputs": inputs, "data_samples": data_samples}


class Classifier(ImageClassifier):
    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: list[DataSample] | None = None,
        mode: str = "tensor",
    ):
        if mode == "tensor":
            feats = self.extract_feat(inputs, data_samples)
            return self.head(feats) if self.with_head else feats # type:ignore
        elif mode == "loss":
            return self.loss(inputs, data_samples)
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs, data_samples, stage="neck"):
        assert stage in ["backbone", "neck", "pre_logits"], (
            f'Invalid output stage "{stage}", please choose from "backbone", '
            '"neck" and "pre_logits"'
        )

        x = self.backbone(inputs, data_samples)

        if stage == "backbone":
            return x
        if self.with_neck:
            x = self.neck(x) # type:ignore
        if stage == "neck":
            return x

        assert self.with_head and hasattr(
            self.head, "pre_logits"
        ), "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor, data_samples: list[DataSample]) -> dict:
        feats = self.extract_feat(inputs, data_samples)
        return self.head.loss(feats, data_samples)

    def predict(
        self,
        inputs: torch.Tensor,
        data_samples: list[DataSample] | None = None,
        **kwargs,
    ) -> list[DataSample]:
        feats = self.extract_feat(inputs, data_samples)
        return self.head.predict(feats, data_samples, **kwargs)


class PackTwoFeats(PackInputs):
    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results["inputs"] = self.format_input(input_)

        data_sample = DataSample()
        data_sample.set_gt_label(results["gt_label"])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type="metainfo")

        packed_results["data_samples"] = data_sample
        return packed_results

# 第一版模型，用于跑通程序以及简单观察可拟合性
class MLP(BaseBackbone):
    def __init__(self, in_channels: int, hidden_channels: list[int], *args, **kwargs):
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
        return (self.layers(x),)


class YuTing_RFSS(BaseBackbone):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 448,
        out_channels: int = 22,
        *args,
        **kwargs,
    ):
        super(YuTing_RFSS, self).__init__(*args, **kwargs)
        self.hidden1 = nn.Linear(
            in_features=in_channels, out_features=hidden_channels, bias=True
        )
        self.hidden2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.hidden3 = nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.hidden4 = nn.Linear(hidden_channels // 4, hidden_channels // 8)
        self.rfss_predict = nn.Linear(hidden_channels // 8, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.activation(self.hidden4(x))
        return (x,)


class YuTing_RFSS_svp(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(YuTing_RFSS_svp, self).__init__()
        self.hidden1 = nn.Linear(
            in_features=in_channels, out_features=hidden_channels, bias=True
        )
        self.hidden2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.hidden3 = nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.hidden_cat = nn.Linear(hidden_channels // 4 + 2, hidden_channels // 8)
        self.activation = nn.ReLU()

    def forward(self, x, svp):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x_cat = torch.cat((x, svp), dim=1)
        x = self.activation(self.hidden_cat(x_cat))
        return (x,)


class YiQin_WeightedPatch(BaseBackbone):
    def __init__(
        self,
        num_CLAM_feats,
        num_heads: int,
        num_CLAM_feat_channel: int = NUM_CLAM_FEAT_CHANNEL,
        out_CLAM_feat_channels: int = 64,
        *args,
        **kwargs,
    ):
        super(YiQin_WeightedPatch, self).__init__(*args, **kwargs)
        self.num_CLAM_feats = num_CLAM_feats
        self.num_heads = num_heads
        self.num_CLAM_feat_channel = num_CLAM_feat_channel

        self.compressor_mha = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=num_CLAM_feat_channel,
                    num_heads=num_heads,
                    batch_first=True,
                ),
                nn.MultiheadAttention(
                    embed_dim=num_CLAM_feat_channel // 4,
                    num_heads=num_heads,
                    batch_first=True,
                ),
                nn.MultiheadAttention(
                    embed_dim=num_CLAM_feat_channel // 16,
                    num_heads=num_heads,
                    batch_first=True,
                ),
            ]
        )
        self.compressor_Linear = nn.ModuleList(
            [
                nn.Linear(num_CLAM_feat_channel, num_CLAM_feat_channel // 4),
                nn.Linear(num_CLAM_feat_channel // 4, num_CLAM_feat_channel // 16),
                nn.Linear(num_CLAM_feat_channel // 16, 1),
            ]
        )
        self.channel_compress = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.csv_extractor = nn.Sequential(
            nn.Linear(NUM_CSV_FEAT, 2 * NUM_CSV_FEAT),
            nn.GELU(),
            nn.Linear(2 * NUM_CSV_FEAT, 4 * NUM_CSV_FEAT),
            nn.GELU(),
            nn.Linear(4 * NUM_CSV_FEAT, 4 * NUM_CSV_FEAT),
            nn.GELU(),
        )
        self.fused_extractor = nn.Sequential(
            nn.Linear(512 + 4 * NUM_CSV_FEAT, out_CLAM_feat_channels * 4),
            nn.GELU(),
            nn.Linear(out_CLAM_feat_channels * 4, out_CLAM_feat_channels * 2),
            nn.GELU(),
            nn.Linear(out_CLAM_feat_channels * 2, out_CLAM_feat_channels),
            nn.GELU(),
        )

    def forward(
        self, inputs: torch.Tensor, data_samples: list[DataSample]
    ) -> torch.Tensor:
        """
        Args:
            CLAM_feat (torch.Tensor): [N, C, 1024]
            csv_feat  (torch.Tensor): [N, 69]
        Returns:
            torch.Tensor: [N, out_CLAM_feat_channels + 69]
        """
        CLAM_feat = inputs
        CSV_feat = torch.stack([i.feat_csv for i in data_samples])

        # patch-wise compression
        # [N, C, 1024] -> [N, 1, C]
        CLAM_compress_weight = CLAM_feat
        for mha, lin in zip(self.compressor_mha, self.compressor_Linear):
            CLAM_compress_weight, _ = mha(
                CLAM_compress_weight, CLAM_compress_weight, CLAM_compress_weight
            )
            CLAM_compress_weight = lin(CLAM_compress_weight)
        # [N, 1, C] * [N, C, 1024] -> [N, 1, 1024]
        patchwise_compressed = torch.matmul(
            CLAM_compress_weight.transpose(1, 2), CLAM_feat
        ).squeeze(1)

        # channel-wise compression
        # [N, 1024] -> [N, out_CLAM_feat_channels]
        channelwise_compressed = self.channel_compress(patchwise_compressed)

        # feature fusion and extraction
        CSV_feat = self.csv_extractor(CSV_feat)
        concat_feat = torch.cat((channelwise_compressed, CSV_feat), dim=1)
        fused_feat = self.fused_extractor(concat_feat)

        return (fused_feat,)


class SVM(BaseBackbone):
    def __init__(self, use_CLAM_feat: bool, use_CSV_feat: bool):
        super().__init__()
        self.use_CLAM_feat = use_CLAM_feat
        self.use_CSV_feat = use_CSV_feat

    def forward(
        self, inputs: torch.Tensor, data_samples: list[DataSample]
    ) -> torch.Tensor:
        """
        Identify projection, the svm's Linear is implemented in head module,
        rather in backbone module.
        """
        feats = []
        if self.use_CLAM_feat:
            feats.append(inputs if inputs.ndim==2 else inputs.mean(dim=1))
        if self.use_CSV_feat:
            feats.append(torch.stack([i.feat_csv for i in data_samples]))
        
        if len(feats) == 2:
            feat = torch.cat(feats, dim=1)
        
        return (feat,)


class Label_for_SVM(BaseTransform):
    def transform(self, results:dict):
        l = results["gt_score"]
        l[l <0.5] = -1
        l[l >=0.5] = 1
        results["gt_score"] = l
        return results
