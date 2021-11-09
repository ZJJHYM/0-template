import torch
import torch.nn as nn
from model.utils import weights_init
from model import ContextGating

from module.base import VideoClassifier


class ResidualMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.mlp_1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_features)
        self.mlp_2 = nn.Linear(hidden_features, out_features)

    def forward(self, inputs):
        x = self.dropout(inputs)
        x = self.mlp_1(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.mlp_2(x)

        return x


class ResidualMixer(nn.Module):
    def __init__(self, patch, channel, hidden_dim, dropout):
        super().__init__()
        self.patch_mlp = ResidualMLP(patch, patch, hidden_dim, dropout)
        self.channel_mlp = ResidualMLP(channel, channel, hidden_dim, dropout)

        self.norm_1 = nn.LayerNorm(channel)
        self.norm_2 = nn.LayerNorm(channel)

    def forward(self, inputs):
        x = inputs
        x = self.norm_1(x)
        x = torch.transpose(x, -2, -1)
        x = self.patch_mlp(x)
        x = torch.transpose(x, -2, -1)
        x = x + inputs
        patch_x = torch.clone(x)

        x = self.norm_2(x)
        x = self.channel_mlp(x)

        return x + patch_x


class StackedResidualMixer(nn.Module):
    def __init__(self, patch, channel, hidden_dim, dropout, layer):
        super().__init__()

        mixer_list = list()
        for _ in range(layer):
            mixer_list.append(ResidualMixer(patch, channel, hidden_dim, dropout))

        self.mixers = nn.ModuleList(mixer_list)

    def forward(self, inputs):
        x = inputs
        for mixer in self.mixers:
            x = mixer(x) + x

        return x


class ResidualMLPPooling(nn.Module):
    def __init__(self, patch, channel, hidden_dim, dropout):
        # pooling patch -> 1
        super().__init__()
        self.patch_mlp = ResidualMLP(patch, 1, hidden_dim, dropout)
        self.channel_mlp = ResidualMLP(channel, channel, hidden_dim, dropout)

        self.norm_1 = nn.LayerNorm(channel)
        self.norm_2 = nn.LayerNorm(channel)

    def forward(self, inputs):
        # inputs -> (batch_size, patch, channel)
        x = inputs

        x = self.norm_1(x)
        x = self.channel_mlp(x)
        x = x + inputs

        x = torch.transpose(x, -2, -1)
        x = self.patch_mlp(x).squeeze()  # (batch_size, channel)
        x = self.norm_2(x)

        return x


class ResidualMLPClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.modal_list = config["modal_list"]

        patch_cnt = 0
        modal_projections = dict()
        for modal_name in self.modal_list:
            patch_cnt += self.config[f"{modal_name}_padding_size"]
            modal_projections[modal_name] = ResidualMLP(
                in_features=self.config[f"{modal_name}_dim"],
                out_features=self.config["projected_dim"],
                hidden_features=self.config["hidden_dim"],
                dropout=self.config["dropout"],
            )

        self.modal_projections = nn.ModuleDict(modal_projections)

        self.mixer_layer = StackedResidualMixer(
            patch_cnt,
            self.config["projected_dim"],
            self.config["hidden_dim"],
            dropout=self.config["dropout"],
            layer=self.config["mixer_layer"],
        )

        self.mlp_pooling = ResidualMLPPooling(
            patch_cnt,
            self.config["projected_dim"],
            self.config["hidden_dim"],
            dropout=self.config["dropout"],
        )

        self.classifier = nn.Sequential(
            ResidualMLP(
                self.config["projected_dim"],
                self.config["projected_dim"],
                self.config["hidden_dim"],
                dropout=self.config["dropout"],
            ),
            ResidualMLP(
                self.config["projected_dim"],
                self.config["num_classes"],
                self.config["hidden_dim"],
                dropout=self.config["dropout"],
            ),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        feature_dict = dict()

        for modal in self.modal_list:
            feature_dict[modal] = self.modal_projections[modal](
                inputs[f"{modal}_feature"].squeeze().type(torch.float)
            )

        feature = torch.cat(list(feature_dict.values()), dim=1)
        del feature_dict

        feature = self.mixer_layer(feature)
        feature = self.mlp_pooling(feature)
        result = self.classifier(feature)

        return {"predicted_label": result}
