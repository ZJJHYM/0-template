import torch
import torch.nn as nn


class ModalContextGatingBlock(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()

        self.gating = nn.Sequential(
            nn.Linear(in_features=dim_b, out_features=dim_a), nn.Sigmoid()
        )
        self.fc = nn.Linear(in_features=dim_a, out_features=dim_a)

    def forward(self, inputs):
        feature_a = inputs["feature_a"]
        feature_b = inputs["feature_b"]

        gating = self.gating(feature_b)
        feature = torch.multiply(feature_a, gating)
        feature = self.fc(feature)

        return feature + feature_a  # residual


class ModalContextGatingLayer(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.cg_block_ab = ModalContextGatingBlock(dim_a=dim_a, dim_b=dim_b)
        self.cg_block_ba = ModalContextGatingBlock(dim_a=dim_b, dim_b=dim_a)

    def forward(self, inputs):
        feature_a = inputs["feature_a"]
        feature_b = inputs["feature_b"]

        return {
            "feature_a": self.cg_block_ab(
                {"feature_a": feature_a, "feature_b": feature_b}
            ),
            "feature_b": self.cg_block_ba(
                {"feature_a": feature_b, "feature_b": feature_a}
            ),
        }


class StackedModalContextGating(nn.Module):
    def __init__(self, dim_a, dim_b, layer):
        super().__init__()
        layer_list = list()
        for _ in range(layer):
            layer_list.append(ModalContextGatingLayer(dim_a, dim_b))

        self.cg_layers = nn.ModuleList(layer_list)

    def forward(self, inputs):
        x = inputs

        for layer in self.cg_layers:
            x = layer(x)

        return x
