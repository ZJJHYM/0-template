import torch
import torch.nn as nn
from model.utils import weights_init
from model import SE
import math
from module.base import VideoClassifier


class FM(nn.Module):
    """Reference: https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/interaction.py
    
    Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1, embedding_size)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(
            torch.sum(fm_input, dim=1, keepdim=True), 2
        )  # (x+y)^2, (batch_size, 1, embedding_size)
        sum_of_square = torch.sum(
            fm_input * fm_input, dim=1, keepdim=True
        )  # x^2 + y^2, (batch_size, 1, embedding_size)
        cross_term = 0.5 * (square_of_sum - sum_of_square)  # xy

        return cross_term


def embedding_augmentation(inputs):
    # inputs -> (batch_size, modal_num, embed_dim)
    x = inputs
    result = list()
    for i in range(inputs.size(1)):
        result.append(inputs[:, i, :])

    for i in range(inputs.size(1)):
        for j in range(i + 1, inputs.size(1)):
            embedding_i = x[:, i, :]
            embedding_j = x[:, j, :]
            result.append(embedding_i * embedding_j)  # (batch_size, embedding_size)

    result = torch.cat(result, dim=-1)

    return result


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerPoolingClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.poolings = nn.ModuleDict()
        self.clses = nn.ParameterDict()

        embed_sum = 0

        for modal in self.config.modal_list:
            embed_sum += self.config[f"{modal}_dim"]
            self.poolings[modal] = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.config[f"{modal}_dim"],
                    nhead=self.config.transformer_head,
                    dropout=self.config.dropout,
                ),
                num_layers=self.config.transformer_layer,
            )
            self.clses[modal] = nn.parameter.Parameter(
                torch.rand(self.config.num_classes, self.config[f"{modal}_dim"]),
                requires_grad=True,
            )

        self.fm = FM()
        self.classifiers = nn.ModuleList()

        for _ in range(self.config.num_classes):
            self.classifiers.append(
                nn.Sequential(
                    nn.Dropout(self.config.dropout),
                    nn.Linear(768 * 6, self.config.hidden_dim),
                    SE(
                        embed_dim=self.config.hidden_dim,
                        hidden_dim=self.config.hidden_dim,
                        dropout=self.config.dropout,
                    ),
                    nn.Linear(self.config.hidden_dim, 1),
                    nn.Sigmoid(),
                )
            )

    def forward(self, inputs):
        feature_dict = dict()

        for modal in self.config.modal_list:
            temp_feature = inputs[f"{modal}_feature"].squeeze().type(torch.float)
            batch_size = temp_feature.size(0)
            temp_cls = torch.stack([self.clses[modal]] * batch_size)
            temp_feature = torch.cat([temp_cls, temp_feature], dim=1)

            temp_feature = torch.transpose(temp_feature, 0, 1)
            feature_dict[modal] = self.poolings[modal](
                temp_feature
            )  # (82, batch_size, modal_dim)

        result = list()
        for idx in range(self.config.num_classes):
            feature = list()
            for modal in self.config.modal_list:
                if modal == "audio":
                    feature.append(
                        torch.cat([feature_dict[modal][idx, :, :]] * 6, dim=-1)
                    )
                else:
                    feature.append(feature_dict[modal][idx, :, :])

            feature = torch.stack(feature)
            feature = torch.transpose(feature, 0, 1)
            feature = embedding_augmentation(feature)

            result.append(self.classifiers[idx](feature))

        result = torch.cat(result, dim=-1)
        return {"predicted_label": result}
