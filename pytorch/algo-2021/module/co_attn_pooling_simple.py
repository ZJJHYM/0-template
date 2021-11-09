import torch
import torch.nn as nn
from model import AttentionPooling, ContextGating, StackedCoAttention
from model.utils import weights_init

from module.base import VideoClassifier


class CoAttnWithAttnPoolingSimpleClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        modal_list = ["video", "text", "audio"]
        self.modal_list = modal_list

        # modal co attention models
        # use a ModuleDict to store
        co_attention_dict = dict()
        for modal in modal_list:
            for other_modal in modal_list:
                if modal == other_modal:
                    continue

                co_attention_dict[f"{modal}_{other_modal}"] = StackedCoAttention(
                    embed_dim=self.config[f"{modal}_dim"],
                    num_heads=self.config["attention_head"],
                    dropout=0.3,
                    kdim=self.config[f"{other_modal}_dim"],
                    layer=self.config["co_attention_layer"],
                )

        self.co_attention_dict = nn.ModuleDict(co_attention_dict)

        # modal self attention models
        # use a ModuleDict to store
        self_attention_dict = dict()
        pooling_dict = dict()
        for modal in modal_list:
            self_attention_dict[modal] = StackedCoAttention(
                embed_dim=self.config[f"{modal}_dim"] * 2,
                num_heads=self.config["attention_head"],
                dropout=0.3,
                kdim=self.config[f"{modal}_dim"] * 2,
                layer=self.config["self_attention_layer"],
            )

            pooling_dict[modal] = AttentionPooling(
                embed_dim=self.config["pooling_dim"],
                num_heads=8,
                dropout=0.3,
                kdim=self.config[f"{modal}_dim"] * 2,
                num_classes=1,
            )

        self.self_attention_dict = nn.ModuleDict(self_attention_dict)
        self.pooling_dict = nn.ModuleDict(pooling_dict)

        # classifiers
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.config["pooling_dim"] * 3, self.config["pooling_dim"]),
            nn.GroupNorm(1, self.config["pooling_dim"]),
            nn.ReLU(),
            ContextGating(self.config["pooling_dim"]),
            nn.Linear(self.config["pooling_dim"], config["num_classes"]),
            nn.Sigmoid(),
        )

        self.classifier.apply(weights_init)

    def forward(self, inputs):
        feature_dict = dict()
        mask_dict = dict()

        for modal in self.modal_list:
            feature_dict[modal] = (
                inputs[f"{modal}_feature"].squeeze().to(self.device).type(torch.float)
            )  # dim=(bs, padding_size, 1024)
            feature_dict[modal] = torch.transpose(feature_dict[modal], 0, 1)
            mask_dict[modal] = inputs[f"{modal}_mask"].to(self.device).type(torch.float)

        # calculate co attention cross modal
        for modal in self.modal_list:
            for other_modal in self.modal_list:
                if modal == other_modal:
                    continue

                feature_name = f"{modal}_{other_modal}"
                feature_dict[feature_name] = self.co_attention_dict[feature_name](
                    feature_dict[modal],
                    feature_dict[other_modal],
                    mask_dict[modal],
                    mask_dict[other_modal],
                )

        for modal in self.modal_list:
            # concat co attention of the same modal
            temp_feature_list = list()
            need_to_pop_list = list()
            for feature_name in feature_dict:
                if f"{modal}_" in feature_name:
                    need_to_pop_list.append(feature_name)
                    temp_feature_list.append(feature_dict[feature_name])

            for feature_name in need_to_pop_list:
                feature_dict.pop(feature_name)

            feature_dict[modal] = torch.cat(temp_feature_list, dim=-1)

            # now calculate self attention of each modal
            feature_dict[modal] = self.self_attention_dict[modal](
                feature_dict[modal],
                feature_dict[modal],
                mask_dict[modal],
                mask_dict[modal],
            )

            # now pooling sequences of features with attn pooling
            feature_dict[modal] = self.pooling_dict[modal](
                feature_dict[modal], mask_dict[modal].type(torch.bool),
            ).squeeze()

        # now concat modal features and feed into classifiers
        feature = torch.cat(list(feature_dict.values()), dim=-1)
        result = self.classifier(feature)

        return {
            "predicted_label": result,
        }
