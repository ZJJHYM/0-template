import torch
import torch.nn as nn
from model.utils import weights_init
from model import SE

from module.base import VideoClassifier


class NaiveClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout=0.3), num_layers=3,
        )

        self.cls = nn.parameter.Parameter(
            torch.rand(self.config.num_classes, 768), requires_grad=True
        )

        self.classifiers = nn.ModuleList()

        for _ in range(self.config.num_classes):
            self.classifiers.append(
                nn.Sequential(
                    nn.Dropout(self.config.dropout),
                    nn.Linear(768, 768),
                    SE(embed_dim=768, hidden_dim=512, dropout=0.3),
                    # nn.Linear(self.config.projected_dim, self.config.num_classes),
                    nn.Linear(512, 1),
                    nn.Sigmoid(),
                )
            )

    def forward(self, inputs):
        feature_dict = dict()

        for modal in self.config.modal_list:
            feature_dict[modal] = inputs[f"{modal}_feature"].squeeze().type(torch.float)

        feature_dict["audio"] = torch.cat([feature_dict["audio"]] * 6, dim=-1)

        feature = torch.cat(
            list(feature_dict.values()), dim=1
        )  # (batch_size, sum(modal_padding_size), projected_dim)

        batch_size = feature.shape[0]
        temp_cls = torch.stack([self.cls] * batch_size)

        feature = torch.cat([temp_cls, feature], dim=1)
        feature = torch.transpose(feature, 0, 1)
        feature = self.pooling(feature)

        result = list()
        for idx in range(self.config.num_classes):
            temp_feature = feature[idx, :, :]
            result.append(self.classifiers[idx](temp_feature))

        result = torch.cat(result, dim=-1)
        return {"predicted_label": result}
