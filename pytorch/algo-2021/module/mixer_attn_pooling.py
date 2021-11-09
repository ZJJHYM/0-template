import torch
import torch.nn as nn
from model import AttentionPooling, ContextGating, StackedMixerLayer
from model.utils import weights_init

from module.base import VideoClassifier


class MixerWithAttnPoolingClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.modal_list = config["modal_list"]

        patch_cnt = 0
        modal_projections = dict()
        for modal_name in self.modal_list:
            modal_projections[modal_name] = nn.Sequential(
                nn.LayerNorm(self.config[f"{modal_name}_dim"]),
                nn.Linear(self.config[f"{modal_name}_dim"], self.config["pooling_dim"]),
            )
            patch_cnt += self.config[f"{modal_name}_padding_size"]
        self.modal_projections = nn.ModuleDict(modal_projections)

        self.mixer_layer = StackedMixerLayer(
            patch=patch_cnt,
            dim=self.config["pooling_dim"],
            layer=self.config["mixer_layer"],
        )

        #         self.attn_pooling = AttentionPooling(
        #             embed_dim=self.config["pooling_dim"],
        #             num_heads=self.config["pooling_head"],
        #             dropout=0.3,
        #             kdim=self.config["pooling_dim"] * 2,
        #             num_classes=self.config["num_classes"],
        #         )

        # classifiers
        #         classifiers = list()
        #         for _ in range(self.config["num_classes"]):
        #             classifiers.append(
        #                 nn.Sequential(
        #                     nn.Dropout(0.3),
        #                     nn.Linear(
        #                         self.config["pooling_dim"] * 2, self.config["pooling_dim"]
        #                     ),
        #                     nn.LayerNorm(self.config["pooling_dim"]),
        #                     nn.ReLU(),
        #                     ContextGating(self.config["pooling_dim"]),
        #                     nn.Linear(self.config["pooling_dim"], 1),
        #                     nn.Sigmoid(),
        #                 )
        #             )

        #         self.classifiers = nn.ModuleList(classifiers)
        #         for classifier in self.classifiers:
        #             classifier.apply(weights_init)

        self.pooling = nn.Sequential(
            nn.Linear(patch_cnt, patch_cnt),
            nn.LayerNorm(patch_cnt),
            nn.ReLU(),
            #             ContextGating(patch_cnt),
            nn.Linear(patch_cnt, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.config["pooling_dim"], self.config["pooling_dim"]),
            nn.LayerNorm(self.config["pooling_dim"]),
            nn.ReLU(),
            ContextGating(self.config["pooling_dim"]),
            nn.Linear(self.config["pooling_dim"], self.config["num_classes"]),
            nn.Sigmoid(),
        )

        self.classifier.apply(weights_init)

    def forward(self, inputs):
        feature_dict = dict()
        mask_dict = dict()

        for modal in self.modal_list:
            temp_feature = inputs[f"{modal}_feature"].squeeze().type(torch.float)
            feature_dict[modal] = self.modal_projections[modal](temp_feature)
            mask_dict[modal] = inputs[f"{modal}_mask"].type(torch.float)

        feature = torch.cat(
            list(feature_dict.values()), dim=1
        )  # (batch_size, max_length * 3, pooling_dim)
        del feature_dict
        # mask = torch.cat(list(mask_dict.values()), dim=-1)
        # del mask_dict

        feature = self.mixer_layer(feature)
        # feature = torch.transpose(feature, 0, 1)
        # feature = self.attn_pooling(feature, None)

        #         result = list()
        #         for idx, classifier in enumerate(self.classifiers):
        #             temp_feature = feature[idx, :]
        #             temp_result = classifier(temp_feature)
        #             result.append(temp_result)

        #         result = torch.cat(result, dim=-1)

        #         feature = torch.mean(feature, dim=1)

        feature = torch.transpose(feature, -2, -1)
        feature = self.pooling(feature).squeeze()
        result = self.classifier(feature)

        return {
            "predicted_label": result,
        }
