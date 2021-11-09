import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.Linear(in_features=dim, out_features=dim),
        )

    def forward(self, inputs):
        return self.model(inputs)


class MixerLayer(nn.Module):
    def __init__(self, patch, dim):
        super().__init__()
        self.patch_mlp = MLP(patch)
        self.channel_mlp = MLP(dim)

        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

    def forward(self, inputs):
        """[summary]

        Args:
            inputs (Tensor): (batch_size, max_length, feature_dim)
        """
        x = self.norm_1(inputs)
        x = torch.transpose(x, -2, -1)  # (batch_size, feature_dim, max_length)
        x = self.patch_mlp(x)
        x = torch.transpose(x, -2, -1)
        x = x + inputs  # residual
        patch_x = torch.clone(x)

        x = self.norm_2(x)
        x = self.channel_mlp(x)

        return x + patch_x  # residual


class StackedMixerLayer(nn.Module):
    def __init__(self, patch, dim, layer):
        super().__init__()

        mixer_list = list()

        for _ in range(layer):
            mixer_list.append(MixerLayer(patch, dim))

        self.mixers = nn.ModuleList(mixer_list)

    def forward(self, inputs):
        x = inputs
        for mixer in self.mixers:
            x = mixer(x)

        return x
