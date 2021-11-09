import torch
import torch.nn as nn
from model import StackedMixerLayer, ContextGating
from model.utils import weights_init

from module.base import VideoClassifier


class MixerClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.modal_list = config["modal_list"]

        patch_cnt = 0
        modal_mixers = dict()
        modal_projections = dict()
        for modal_name in self.modal_list:
            modal_mixers[modal_name] = StackedMixerLayer(
                patch=self.config[f"{modal_name}_padding_size"],
                dim=self.config[f"{modal_name}_dim"],
                layer=self.config["self_mixer_layer"],
            )
            modal_projections[modal_name] = nn.Sequential(
                nn.Linear(
                    self.config[f"{modal_name}_dim"], self.config["projected_dim"]
                ),
                ContextGating(self.config["projected_dim"]),
            )
            patch_cnt += self.config[f"{modal_name}_padding_size"]

        self.modal_mixers = nn.ModuleDict(modal_mixers)
        self.modal_projections = nn.ModuleDict(modal_projections)

        self.full_mixer_layer = StackedMixerLayer(
            patch=patch_cnt,
            dim=self.config["projected_dim"],
            layer=self.config["full_mixer_layer"],
        )

        self.mlp_pooling = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(patch_cnt, self.config["projected_dim"] * 2),
            nn.GELU(),
            nn.LayerNorm(self.config["projected_dim"] * 2),
            ContextGating(self.config["projected_dim"] * 2),
            nn.Linear(self.config["projected_dim"] * 2, self.config["projected_dim"]),
            nn.GELU(),
            nn.LayerNorm(self.config["projected_dim"]),
            ContextGating(self.config["projected_dim"]),
            nn.Linear(self.config["projected_dim"], 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            ContextGating(self.config["projected_dim"]),
            nn.Linear(self.config["projected_dim"], self.config["projected_dim"] * 2),
            nn.GELU(),
            nn.LayerNorm(self.config["projected_dim"] * 2),
            nn.Linear(self.config["projected_dim"] * 2, self.config["projected_dim"]),
            nn.GELU(),
            nn.LayerNorm(self.config["projected_dim"]),
            ContextGating(self.config["projected_dim"]),
            nn.Linear(self.config["projected_dim"], self.config["num_classes"]),
            nn.Sigmoid(),
        )

        self.mlp_pooling.apply(weights_init)
        self.classifier.apply(weights_init)

    def forward(self, inputs):
        feature_dict = dict()

        for modal in self.modal_list:
            feature_dict[modal] = inputs[f"{modal}_feature"].squeeze().type(torch.float)
            feature_dict[modal] = self.modal_mixers[modal](feature_dict[modal])
            feature_dict[modal] = self.modal_projections[modal](feature_dict[modal])

        feature = torch.cat(
            list(feature_dict.values()), dim=1
        )  # (batch_size, patch_cnt, projected_dim)
        del feature_dict

        feature = torch.transpose(
            feature, -2, -1
        )  # (batch_size, projected_dim, patch_cnt)

        feature = self.mlp_pooling(feature).squeeze()
        result = self.classifier(feature)

        return {
            "predicted_label": result,
        }
