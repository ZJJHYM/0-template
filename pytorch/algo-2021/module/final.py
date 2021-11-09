import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import weights_init
from model import ContextGating

from module.base import VideoClassifier


class Attention(nn.Module):
    def __init__(self, d_model, d_value, d_hidden, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            kdim=d_value,
            vdim=d_value,
        )

        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)

        self.activation = nn.ReLU()

    def forward(self, q, v=None, key_padding_mask=None, attn_mask=None):
        q2, _ = self.attn(
            q, v, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        q = q + self.dropout(q2)
        q = self.norm1(q)
        q2 = self.linear2(self.dropout(self.activation(self.linear1(q))))
        q = q + self.dropout2(q)
        q = self.norm2(q)

        return q


class StackedAttention(nn.Module):
    def __init__(self, d_model, d_value, d_hidden, nhead, dropout, layer):
        super().__init__()

        self.layers = nn.ModuleList()
        self.nhead = nhead
        for _ in range(layer):
            self.layers.append(
                Attention(
                    d_model=d_model,
                    d_value=d_value,
                    d_hidden=d_hidden,
                    nhead=nhead,
                    dropout=dropout,
                )
            )

    def forward(self, q, v=None, q_mask=None, v_mask=None, is_self_attention=False):
        assert (v is not None) or is_self_attention

        if not is_self_attention and q_mask is not None and v_mask is not None:
            attn_mask = torch.matmul(
                q_mask.type(torch.float).unsqueeze(2),
                v_mask.type(torch.float).unsqueeze(1),
            )
            attn_mask = torch.cat([attn_mask] * self.nhead, dim=0).type(torch.bool)
            v_mask = v_mask.type(torch.bool)

        elif is_self_attention and q_mask is not None:
            attn_mask = torch.matmul(
                q_mask.type(torch.float).unsqueeze(2),
                q_mask.type(torch.float).unsqueeze(1),
            )
            attn_mask = torch.cat([attn_mask] * self.nhead, dim=0).type(torch.bool)
            q_mask = q_mask.type(torch.bool)
        else:
            attn_mask = None

        for layer in self.layers:
            if is_self_attention:
                q = layer(q, q, q_mask, None)
            else:
                q = layer(q, v, v_mask, None)

        return q


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, kdim, num_classes):
        super().__init__()
        self.q = nn.parameter.Parameter(torch.rand(num_classes, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=kdim,
            vdim=kdim,
        )

        self.layer_norm_final = nn.LayerNorm(embed_dim)

    def forward(self, feature, mask):
        # expand self.q to batch size
        batch_size = feature.shape[1]
        expanded_q = torch.stack(
            [self.q] * batch_size, dim=0
        )  # (N, num_classes, embed_dim)
        expanded_q = torch.transpose(expanded_q, 0, 1)  # (num_classes, N, embed_dim)

        # (num_classes, N, embed_dim)
        pooled_inputs, _ = self.attn(
            expanded_q, feature, feature, key_padding_mask=mask
        )
        pooled_inputs = self.layer_norm_final(pooled_inputs)
        return pooled_inputs  # (num_classes, N, embed_dim)


class FinalClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        # modal co attention models
        # use a ModuleDict to store
        self.co_attention_dict = nn.ModuleDict()
        for modal in self.config.modal_list:
            for other_modal in self.config.modal_list:
                if modal == other_modal:
                    continue

                self.co_attention_dict[f"{modal}_{other_modal}"] = StackedAttention(
                    d_model=self.config[f"{modal}_dim"],
                    d_value=self.config[f"{other_modal}_dim"],
                    d_hidden=self.config.attention_hidden_dim,
                    nhead=self.config.attention_head,
                    dropout=self.config.attention_dropout,
                    layer=self.config.co_attention_layer,
                )

        # modal self attention models
        # use a ModuleDict to store
        self.self_attention_dict = nn.ModuleDict()
        self.pooling_dict = nn.ModuleDict()
        self.feature_aug = nn.ModuleDict()
        for modal in self.config.modal_list:
            self.self_attention_dict[modal] = StackedAttention(
                d_model=self.config[f"{modal}_dim"] * (len(self.config.modal_list) - 1),
                d_value=self.config[f"{modal}_dim"] * (len(self.config.modal_list) - 1),
                d_hidden=self.config.attention_hidden_dim,
                nhead=self.config.attention_head,
                dropout=self.config.attention_dropout,
                layer=self.config.self_attention_layer,
            )

            self.pooling_dict[modal] = AttentionPooling(
                embed_dim=self.config.pooling_dim,
                num_heads=self.config.pooling_head,
                dropout=self.config.attention_dropout,
                kdim=self.config[f"{modal}_dim"] * (len(self.config.modal_list) - 1),
                num_classes=self.config.num_classes,
            )

            self.feature_aug[modal] = nn.Sequential(
                ContextGating(self.config.pooling_dim),
                nn.Linear(self.config.pooling_dim, self.config.pooling_dim),
                nn.ReLU(),
                nn.LayerNorm(self.config.pooling_dim),
            )

        # classifiers
        classifiers = list()
        for _ in range(self.config.num_classes):
            classifiers.append(
                nn.Sequential(
                    nn.Dropout(self.config.classifier_dropout),
                    nn.Linear(
                        self.config.pooling_dim * len(self.config.modal_list),
                        self.config.pooling_dim,
                    ),
                    nn.LayerNorm(self.config.pooling_dim),
                    nn.ReLU(),
                    nn.Linear(self.config.pooling_dim, 1),
                    nn.Sigmoid(),
                )
            )

        self.classifiers = nn.ModuleList(classifiers)
        for classifier in self.classifiers:
            classifier.apply(weights_init)

    def forward(self, inputs):
        feature_dict = dict()
        mask_dict = dict()

        for modal in self.config.modal_list:
            feature_dict[modal] = (
                inputs[f"{modal}_feature"].squeeze().type(torch.float)
            )  # dim=(bs, padding_size, 1024)
            feature_dict[modal] = torch.transpose(feature_dict[modal], 0, 1)
            mask_dict[modal] = inputs[f"{modal}_mask"].type(torch.float)

        # calculate co attention cross modal
        for modal in self.config.modal_list:
            for other_modal in self.config.modal_list:
                if modal == other_modal:
                    continue

                feature_name = f"{modal}_{other_modal}"
                feature_dict[feature_name] = self.co_attention_dict[feature_name](
                    feature_dict[modal],
                    feature_dict[other_modal],
                    mask_dict[modal],
                    mask_dict[other_modal],
                )

        for modal in self.config.modal_list:
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

            feature_dict[modal] = self.feature_aug[modal](feature_dict[modal])

        # now concat modal features and feed into classifiers
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

    def training_step(self, batch, batch_idx):
        result = self.forward(batch)
        result_ = self.forward(batch)

        predicted_label = result["predicted_label"]
        gt_label = batch["label"]
        gt_label = (
            torch.transpose(torch.stack(gt_label), 0, 1)
            .type(torch.float)
            .to(self.device)
        )

        ce_loss = self.loss(gt_label, predicted_label)
        kl_loss = self._compute_kl_loss(
            result["predicted_label"], result_["predicted_label"]
        )

        loss_value = ce_loss + self.config.alpha * kl_loss
        self.log("train_loss", loss_value, prog_bar=True)

        train_metrics = self._metric_calculator(gt_label, predicted_label)
        for item in train_metrics:
            self.log(f"train_{item}", train_metrics[item], prog_bar=True)

        if math.isnan(loss_value):
            return None

        return loss_value
