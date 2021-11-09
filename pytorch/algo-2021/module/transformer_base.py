import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import weights_init

import math
import random

from module.base import VideoClassifier


class ContextGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ctx_gate_w = nn.Linear(in_features=dim, out_features=dim)

        nn.init.xavier_normal_(self.ctx_gate_w.weight)

    def forward(self, inputs):
        gated = self.ctx_gate_w(inputs)
        gated = nn.Sigmoid()(gated)

        return torch.mul(gated, inputs)


class TransformerClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.projection = nn.ModuleDict()
        self.cls = nn.ParameterDict()
        self.pooling = nn.ModuleDict()
        self.feature_aug = nn.ModuleDict()

        total_dim = 0

        for modal in self.config.modal_list:
            self.projection[modal] = nn.Sequential(
                nn.Linear(self.config[f"{modal}_dim"], self.config.d_model, bias=False),
                nn.ReLU(),
                nn.LayerNorm(self.config.d_model),
            )
            self.cls[modal] = nn.parameter.Parameter(
                torch.rand(1, self.config.d_model), requires_grad=True
            )
            self.pooling[modal] = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.config.d_model, nhead=self.config.transformer_head,
                ),
                num_layers=self.config.transformer_layer,
            )
            # self.feature_aug[modal] = nn.Sequential(
            #     ContextGating(self.config.d_model),
            #     nn.Linear(self.config.d_model, self.config.d_model),
            #     nn.ReLU(),
            #     nn.LayerNorm(self.config.d_model),
            # )
            total_dim += self.config.d_model

            # self.projection[modal].apply(weights_init)
            # self.feature_aug[modal].apply(weights_init)

        self.classifier = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(total_dim, self.config.num_classes),
            nn.Sigmoid(),
        )

        # weight init
        self.classifier.apply(weights_init)

        # self.classifier = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Dropout(self.config.dropout),
        #             nn.Linear(total_dim, 1),
        #             nn.Sigmoid(),
        #         )
        #         for _ in range(self.config.num_classes)
        #     ]
        # )

    def forward(self, inputs):
        feature_dict = dict()
        mask_dict = dict()

        for modal in self.config.modal_list:
            feature_dict[modal] = (
                inputs[f"{modal}_feature"].squeeze().type(torch.float)
            )  # (batch_size, padding_size, dim)
            feature_dict[modal] = torch.transpose(feature_dict[modal], 0, 1)
            mask_dict[modal] = inputs[f"{modal}_mask"].type(torch.float)

        # random drop some modal according to modal_dropout
        dropout_cnt = 0
        dropout_list = list()
        for modal in self.config.modal_list:
            if (
                dropout_cnt < (len(self.config.modal_list) - 2)
                and random.random() < self.config.modal_dropout
            ):
                # feature_dict[modal] = torch.zeros(
                #     (
                #         feature_dict[modal].size(0),
                #         feature_dict[modal].size(1),
                #         self.config.d_model,
                #     )
                # ).type_as(feature_dict[modal])
                feature_dict[modal] = torch.zeros_like(feature_dict[modal])
                mask_dict[modal] = torch.ones_like(mask_dict[modal])
                dropout_cnt += 1

                dropout_list.append(modal)

        for modal in self.config.modal_list:
            feature_dict[modal] = self.projection[modal](feature_dict[modal])

            batch_size = feature_dict[modal].size(1)
            cls_token = torch.stack([self.cls[modal]] * batch_size)
            cls_token = torch.transpose(cls_token, 0, 1)

            temp_feature = torch.cat([cls_token, feature_dict[modal]], dim=0)
            # temp_mask = torch.cat(
            #     [
            #         torch.zeros((batch_size, 1)).type_as(mask_dict[modal]),
            #         mask_dict[modal],
            #     ],
            #     dim=-1,
            # )

            # feature_dict[modal] = self.pooling[modal](
            #     temp_feature, src_key_padding_mask=temp_mask.type(torch.bool)
            # )
            feature_dict[modal] = self.pooling[modal](temp_feature)

            feature_dict[modal] = feature_dict[modal][0, :, :]

            # feature_dict[modal] = self.feature_aug[modal](feature_dict[modal])

        feature = torch.cat(list(feature_dict.values()), dim=-1)

        result = self.classifier(feature)
        # result = list()
        # for classifier in self.classifier:
        #     result.append(classifier(feature))
        # result = torch.cat(result, dim=-1)

        return {
            "predicted_label": result,
        }

    def _compute_kl_loss(self, p, q):
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none"
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none"
        )

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss


#     def training_step(self, batch, batch_idx):
#         result = self.forward(batch)
#         result_ = self.forward(batch)

#         predicted_label = result["predicted_label"]
#         gt_label = batch["label"]
#         gt_label = (
#             torch.transpose(torch.stack(gt_label), 0, 1)
#             .type(torch.float)
#             .to(self.device)
#         )

#         ce_loss = self.loss(gt_label, predicted_label)
#         kl_loss = self._compute_kl_loss(
#             result["predicted_label"], result_["predicted_label"]
#         )

#         loss_value = ce_loss + self.config.alpha * kl_loss
#         self.log("train_loss", loss_value, prog_bar=True)

#         train_metrics = self._metric_calculator(gt_label, predicted_label)
#         for item in train_metrics:
#             self.log(f"train_{item}", train_metrics[item], prog_bar=True)

#         if math.isnan(loss_value):
#             return None

#         return loss_value
