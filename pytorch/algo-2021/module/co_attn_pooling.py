import torch
import torch.nn as nn
from model import AttentionPooling, ContextGating, StackedCoAttention, SE
from model.utils import weights_init

from module.base import VideoClassifier


class CoAttnWithAttnPoolingClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        # modal_list = ["video", "text", "audio"]
        modal_list = config["modal_list"]
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
                    dropout=self.config["dropout"],
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
                embed_dim=self.config[f"{modal}_dim"] * (len(self.modal_list) - 1),
                num_heads=self.config["attention_head"],
                dropout=self.config["dropout"],
                kdim=self.config[f"{modal}_dim"] * (len(self.modal_list) - 1),
                layer=self.config["self_attention_layer"],
            )

            pooling_dict[modal] = AttentionPooling(
                embed_dim=self.config["pooling_dim"],
                num_heads=self.config["pooling_head"],
                dropout=self.config["dropout"],
                kdim=self.config[f"{modal}_dim"] * (len(self.modal_list) - 1),
                num_classes=self.config["num_classes"],
            )

        self.self_attention_dict = nn.ModuleDict(self_attention_dict)
        self.pooling_dict = nn.ModuleDict(pooling_dict)

        # classifiers
        classifiers = list()
        for _ in range(self.config["num_classes"]):
            classifiers.append(
                nn.Sequential(
                    nn.Dropout(self.config["dropout"]),
                    nn.Linear(
                        self.config["pooling_dim"] * len(self.modal_list),
                        self.config["pooling_dim"],
                    ),
                    SE(
                        embed_dim=self.config.pooling_dim,
                        hidden_dim=self.config.pooling_dim,
                        dropout=self.config.dropout,
                    ),
                    nn.Linear(self.config["pooling_dim"], 1),
                    nn.Sigmoid(),
                )
            )

        self.classifiers = nn.ModuleList(classifiers)
        for classifier in self.classifiers:
            classifier.apply(weights_init)

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
            )

        # now concat modal features and feed into classifiers
        # TODO: make sure no more keys are in the feature_dict, or this will be wrong
        feature = torch.cat(list(feature_dict.values()), dim=-1)

        result = list()
        for idx, classifier in enumerate(self.classifiers):
            temp_feature = feature[idx, :, :]
            temp_result = classifier(temp_feature)
            result.append(temp_result)

        result = torch.cat(result, dim=-1)

        return {
            "predicted_label": result,
        }
