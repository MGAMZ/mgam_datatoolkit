import os
import pdb
import math
from collections.abc import Callable
from typing_extensions import Literal

import numpy as np
import torch
from torch import nn
from torch import Tensor

from mmcv.transforms import BaseTransform
from mmengine.model import BaseModule
from mmengine.model.base_model import BaseDataPreprocessor
from mmengine.evaluator.metric import BaseMetric
from mmpretrain.registry import MODELS
from mmpretrain.datasets.transforms.formatting import PackInputs, DataSample
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.models.classifiers import ImageClassifier, BaseClassifier


NUM_CSV_FEAT = 68
NUM_CLAM_FEAT_CHANNEL = 1024

# 有两列是中文标注，映射到数字索引
TISSUE_CLASS_INDEX = {
    "default": 0,
    "乳头型": 1,
    "实性型": 2,
    "贴壁型": 3,
    "微乳头型": 4,
    "腺泡型": 5,
}
COMBINATION_LABELS = {
    (12,14): "标签12+14",
    (12,16): "标签12+16",
    (14,16): "标签14+16",
    (12,14,16): "标签12+14+16",
}
EXCLUDE_CLASSES = [
    "date_dx",
    "date_lfp",
    "date_relapse",
    "date_mt",
    "id",
    "蜡块号",
    *COMBINATION_LABELS.values()
]
# 定义标签序号和他们隶属的相关组别
LABEL_GROUP = {
    "Immunization": [1,2,3,4,5,6,7],        # 免疫相关标注
    "Histological": [8,],                   # 组织学形态
    "TumorNucleus": [9,10,11,12,13,14],     # 肿瘤细胞核
    "TumorStroma":  [15,16,],               # 肿瘤间质
    "GenePhenoTp":  [17,18,],               # 基因表型
}


class PathologyPreprocessor(BaseDataPreprocessor):
    def __init__(self, *args, **kwargs):
        super(PathologyPreprocessor, self).__init__(*args, **kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        inputs = torch.stack(self.cast_data(data["inputs"]))
        data_samples = data.get("data_samples", None)
        data_samples = self.cast_data(data_samples)
        return {"inputs": inputs, "data_samples": data_samples}


class Classifier(ImageClassifier):
    def forward(
        self,
        inputs: Tensor,
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

    def loss(self, inputs: Tensor, data_samples: list[DataSample]) -> dict:
        feats = self.extract_feat(inputs, data_samples)
        return self.head.loss(feats, data_samples)

    def predict(
        self,
        inputs: Tensor,
        data_samples: list[DataSample] | None = None,
        **kwargs,
    ) -> list[DataSample]:
        feats = self.extract_feat(inputs, data_samples)
        return self.head.predict(feats, data_samples, **kwargs)

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

    def forward(self, x: Tensor) -> Tensor:
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
        self, inputs: Tensor, data_samples: list[DataSample]
    ) -> Tensor:
        """
        Args:
            CLAM_feat (Tensor): [N, C, 1024]
            csv_feat  (Tensor): [N, 69]
        Returns:
            Tensor: [N, out_CLAM_feat_channels + 69]
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
        self, inputs: Tensor, data_samples: list[DataSample]
    ) -> Tensor:
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


class GrouppedClser(BaseClassifier):
    def __init__(
        self, 
        enable_clam_feat:bool,
        shared:dict, 
        clser_Immunization:dict,
        clser_Histological:dict,
        clser_TumorNucleus:dict,
        clser_TumorStroma:dict,
        clser_GenePhenoTp:dict,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.enable_clam_feat = enable_clam_feat
        if enable_clam_feat:
            self.shared = MODELS.build(shared)
        self.clser_Immunization = MODELS.build(clser_Immunization)
        self.clser_Histological = MODELS.build(clser_Histological)
        self.clser_TumorNucleus = MODELS.build(clser_TumorNucleus)
        self.clser_TumorStroma = MODELS.build(clser_TumorStroma)
        self.clser_GenePhenoTp = MODELS.build(clser_GenePhenoTp)
    
    def forward(self,
                inputs: Tensor,
                data_samples: list[DataSample],
                mode: str = 'tensor'
    ):
        if self.enable_clam_feat:
            feat = self.extract_feat(inputs)
        else:
            feat = None
        
        
        if mode == "tensor":
            return feat

        if mode == "predict":
            with torch.inference_mode():
                results = self.predict(feat, data_samples)
                for i, data_sample in enumerate(data_samples):
                    for sub_group in LABEL_GROUP.keys():
                        if sub_group in data_sample.gt_label.keys():
                            data_sample.set_field(results[sub_group][i], f"pred_label/{sub_group}") 
            return data_samples
        
        if mode == "loss":
            results = self.loss(feat, data_samples)
            losses = {}
            for sub_group in LABEL_GROUP.keys():
                # SampleWise Sum
                result = results.get(sub_group, None)
                if result is None:
                    continue
                # Parse loss when available
                if result.get("loss", None) is not None:
                    losses[f"loss/{sub_group}"] = result["loss"].mean()
                # key without `loss` will not be used to calculate loss
                if result.get("acc", None) is not None:
                    losses[f"acc/{sub_group}"] = result["acc"].mean()
            return losses

    def extract_feat(self, inputs: Tensor):
        return self.shared(inputs)
    
    def loss(self, 
             feat, 
             data_samples: list[DataSample]):
        """
        Args:
            feat: [N, C]
            data_samples (list[DataSample]): [N]
                - gt_label (dict):
                    - Immunization (Tensor): [num_targets]
                    - Histological (Tensor): [num_targets]
                    - TumorNucleus (Tensor): [num_targets]
                    - TumorStroma  (Tensor): [num_targets]
                    - GenePhenoTp  (Tensor): [num_targets]
        
        Returns:
            results (dict):
                - Immunization (list): length N, maybe dict or None
                - Histological (list): length N, maybe dict or None
                - TumorNucleus (list): length N, maybe dict or None
                - TumorStroma  (list): length N, maybe dict or None
                - GenePhenoTp  (list): length N, maybe dict or None
                    (If it is a dict)
                    - loss (Tensor): [1]
                    - acc  (Tensor): [1]
        """
        
        # 先为每个子分组预先分配结果
        results = {}

        for sg in LABEL_GROUP.keys():
            # 收集包含此子目标组的样本索引
            sg_indices = []
            for i, ds in enumerate(data_samples):
                if sg in ds.gt_label.keys():
                    sg_indices.append(i)
            if not sg_indices:
                continue

            # 批量收集
            sub_feats = feat[sg_indices] if feat is not None else None
            sub_feat_annos = torch.stack([data_samples[i].feat_anno 
                                          for i in sg_indices])
            sub_labels = torch.stack([data_samples[i].gt_label[sg] 
                                      for i in sg_indices])
            
            # 调用对应分类器
            clser: BaseModule = getattr(self, f"clser_{sg}")
            batch_results = clser.loss(sub_feats, sub_feat_annos, sub_labels)
            
            results[sg] = batch_results
            
        return results
    
    def predict(self, 
                feat: Literal["loss", "predict"], 
                data_samples: list[DataSample]):
        """
        Args:
            feat (Tensor): [N, C]
            data_samples (list[DataSample]): [N]
                - gt_label (dict):
                    - Immunization (Tensor): [num_targets]
                    - Histological (Tensor): [num_targets]
                    - TumorNucleus (Tensor): [num_targets]
                    - TumorStroma  (Tensor): [num_targets]
                    - GenePhenoTp  (Tensor): [num_targets]
        
        Returns:
            results (dict):
                - Immunization (list): length N, maybe dict or None
                - Histological (list): length N, maybe dict or None
                - TumorNucleus (list): length N, maybe dict or None
                - TumorStroma  (list): length N, maybe dict or None
                - GenePhenoTp  (list): length N, maybe dict or None
                    (If it is a dict)
                    - loss (Tensor): [1]
                    - acc  (Tensor): [1]
        """
        # 先为每个子分组预先分配结果
        results = {}

        for sg in LABEL_GROUP.keys():
            # 收集包含此子分组的样本索引
            sg_indices = []
            for i, ds in enumerate(data_samples):
                if ds.get("feat_anno") is not None:
                    sg_indices.append(i)
            if not sg_indices:
                continue

            # 批量收集
            sub_feats = feat[sg_indices] if feat is not None else None
            sub_feat_annos = torch.stack([data_samples[i].feat_anno 
                                          for i in sg_indices])
            
            # 调用对应分类器
            clser: BaseModule = getattr(self, f"clser_{sg}")
            batch_results = clser.predict(sub_feats, sub_feat_annos)
            
            results[sg] = batch_results
            
        return results


class SharedExtractor1D(BaseBackbone):
    def __init__(self, in_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.proj = nn.ModuleList([
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        ])
    
    def forward(self, inputs: Tensor):
        i = inputs
        for layer in self.proj:
            i = layer(i)
        return i


class SharedExtractor2D(BaseBackbone):
    def __init__(self, 
                 in_n_feats:int, 
                 hidden_channels:list[int], 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.is_perfect_square(in_n_feats)
        
        self.in_n_feats = in_n_feats
        self.h_chans = hidden_channels
        self.feat_2d_size = int(math.sqrt(in_n_feats))
        
        self.layer = nn.ModuleList()
        for i in range(len(self.h_chans)-1):
            group = [
                nn.Conv2d(self.h_chans[i], self.h_chans[i+1], kernel_size=3),
                nn.GELU()
            ]
            self.layer.extend(group)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def is_perfect_square(n):
        if n < 0:
            return False  # 负数没有实数平方根
        root = math.sqrt(n)
        return int(root + 0.5) ** 2 == n
    
    def forward(self, inputs: Tensor):
        # [N, n_feats, C] -> [N, C, n_feats]
        i = inputs.transpose(1, 2)
        # [N, C, n_feats] -> [N, C, feat_2d_size, feat_2d_size]
        i = i.reshape(*i.shape[:2], self.feat_2d_size, self.feat_2d_size)
        for layer in self.layer:
            i = layer(i)
        i = self.pool(i).flatten(1)
        return i  # [N, C]


class SubGroupHead(BaseModule):
    def __init__(self, 
                 num_classes:tuple[int],
                 in_clam_channels:int,
                 enable_clam_feat:bool=True,
                 enable_anno_feat:bool=True,
                 in_anno_channels:int=69,
                 *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.enable_clam_feat = enable_clam_feat
        self.enable_anno_feat = enable_anno_feat
        self.num_classes = num_classes
        in_channels = 0
        if enable_clam_feat:
            in_channels += in_clam_channels
        if enable_anno_feat:
            in_channels += in_anno_channels
        
        self.union_proj = nn.ModuleList([
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        ])
        for i, c in enumerate(num_classes):
            setattr(self, f"target_proj_{i}", nn.ModuleList([
                nn.Linear(in_channels, in_channels//2),
                nn.GELU(),
                nn.Linear(in_channels//2, c),
            ]))
        self.cri = nn.CrossEntropyLoss()

    def forward(self, inputs:Tensor|None=None, anno:Tensor|None=None):
        """
        Args:
            inputs (Tensor): [N, C]
            anno (Tensor): [N, feat_anno_channels]
        Returns:
            pred_logits (list[Tensor]): [N, targets, classes]
        """
        if inputs is not None and inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if anno is not None and anno.ndim == 1:
            anno = anno.unsqueeze(0)
        if self.enable_clam_feat and self.enable_anno_feat:
            assert inputs is not None and anno is not None
            feat = torch.cat((inputs, anno), dim=1)
        elif self.enable_clam_feat:
            assert inputs is not None
            feat = inputs
        elif self.enable_anno_feat:
            assert anno is not None
            feat = anno
        
        union_feat = feat
        for layer in self.union_proj:
            union_feat = layer(union_feat)
        
        logits = []
        for i in range(len(self.num_classes)):
            proj = getattr(self, f"target_proj_{i}")
            feat = union_feat
            for layer in proj:
                feat = layer(feat)
            logits.append(feat)
        
        return logits

    def loss(self, inputs:Tensor|None, anno:Tensor|None, gt_label:Tensor):
        """
        Args:
            inputs (Tensor): [N, C]
            anno (Tensor): [N, feat_anno_channels]
            gt_label (Tensor): [N, targets]
        
        Returns:
            loss (Tensor): [1]
            acc  (Tensor): [1]
        """
        if inputs is not None and inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if anno is not None and anno.ndim == 1:
            anno = anno.unsqueeze(0)
        if gt_label.ndim == 1:
            gt_label = gt_label.unsqueeze(0)
        
        # [num_targets, N, classes]
        pred_logits:list[Tensor] = self.forward(inputs, anno)
        
        losses = []
        accs = []
        for i, logits in enumerate(pred_logits):
            # logits: [N, classes]
            # gt_label: [N, targets]
            loss_one = self.cri(logits, gt_label[:, i])
            losses.append(loss_one)
        
            with torch.inference_mode():
                acc = (logits.argmax(dim=-1) == gt_label[:, i]).float().mean()
                accs.append(acc)
        
        return {"loss": torch.stack(losses).mean(), 
                "acc": torch.stack(accs).mean()}
    
    def predict(self, inputs:Tensor|None, anno:Tensor|None, *args, **kwargs):
        if inputs is not None and inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if anno is not None and anno.ndim == 1:
            anno = anno.unsqueeze(0)
        
        # [num_targets, N, classes]
        pred_logits = self.forward(inputs, anno)
        pred_label = []
        for i, logits in enumerate(pred_logits):
            pred_label.append(logits.argmax(dim=-1))
        # [N, num_targets]
        return torch.stack(pred_label, dim=-1)


class SubGroupMetric(BaseMetric):
    def process(self, data_batch: dict, data_samples: list[dict]) -> None:
        losses = {k: [] for k in LABEL_GROUP.keys()}
        classwise_counts = {k: {} for k in LABEL_GROUP.keys()}
        for data_sample in data_samples:
            for sub_group in LABEL_GROUP.keys():
                if sub_group in data_sample["gt_label"].keys():
                    gt_label = data_sample["gt_label"][sub_group]
                    pred_label = data_sample[f"pred_label/{sub_group}"]
                    acc = (gt_label == pred_label)
                    losses[sub_group].append(acc.cpu().numpy())
                    
                    # Class-Wise 统计
                    for i in range(len(gt_label)):
                        if i not in classwise_counts[sub_group]:
                            classwise_counts[sub_group][i] = [0, 0]  # [correct, total]
                        correct = (pred_label[i].item() == gt_label[i].item())
                        classwise_counts[sub_group][i][0] += int(correct)
                        classwise_counts[sub_group][i][1] += 1
                    
        self.results.append((losses, classwise_counts))

    def compute_metrics(self, results: list) -> dict:
        # ...existing code...
        avg_loss = {k: [] for k in LABEL_GROUP.keys()}
        aggregated_classwise_counts = {k: {} for k in LABEL_GROUP.keys()}
        for (loss_dict, cls_dict) in results:
            for k, v in loss_dict.items():
                avg_loss[k].extend(v)
            # 聚合 Class-Wise 统计
            for sg, class_map in cls_dict.items():
                for class_idx, cnts in class_map.items():
                    if class_idx not in aggregated_classwise_counts[sg]:
                        aggregated_classwise_counts[sg][class_idx] = [0, 0]
                    aggregated_classwise_counts[sg][class_idx][0] += cnts[0]
                    aggregated_classwise_counts[sg][class_idx][1] += cnts[1]

        avg_loss = {f"Acc/{k}": np.mean(v) for k, v in avg_loss.items()}
        avg_loss["Acc/Avg"] = np.mean(list(avg_loss.values()))
        
        # 计算并添加 Class-Wise Acc
        for sg, class_map in aggregated_classwise_counts.items():
            for class_idx, cnts in class_map.items():
                total = cnts[1] if cnts[1] > 0 else 1
                avg_loss[f"ClassWiseAcc/{sg}_{class_idx}"] = cnts[0] / total

        return avg_loss

