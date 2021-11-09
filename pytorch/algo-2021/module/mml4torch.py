import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from module.base import VideoClassifier

CLS = "CLS"
SEP = "SEP"
EOS = "EOS"


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, attention_dim):
        super().__init__()

        self.W = nn.parameter.Parameter(
            torch.rand(embed_dim, attention_dim), requires_grad=True
        )
        self.b = nn.parameter.Parameter(torch.rand(attention_dim), requires_grad=True)
        self.u = nn.parameter.Parameter(
            torch.rand(attention_dim, 1), requires_grad=True
        )

    def forward(self, feature, mask=None):
        # feature -> (sequence_length, batch_size, embed_dim)
        x = feature

        et = torch.add(
            torch.matmul(x, self.W), self.b
        )  # (sq_len, batch_size, attention_dim)
        at = torch.matmul(et, self.u).squeeze()  # (sq_len, batch_size)

        if mask is not None:
            at *= mask.type(torch.float)

        ot = at.unsqueeze(-1) * x
        output = torch.sum(ot, dim=0)

        return output


class AttentionPoolingClassifier(nn.Module):
    def __init__(self, embed_dim, dropout, use_residual=False):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        self.attn_pooling = AttentionPooling(
            embed_dim=embed_dim, attention_dim=embed_dim
        )
        self.classifier = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.use_residual:
            self.norm2 = nn.LayerNorm(768, eps=1e-6)

    def forward(self, inputs, mask=None):
        x = inputs

        if self.use_residual:
            pass
            # short_cut = self.norm2(short_cut)
            # residual = x + short_cut
        else:
            residual = x

        x = self.attn_pooling(residual, mask)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


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


class VideoTransformer(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.token_dict = {
            "CLS": torch.tensor([0]),
            "SEP": torch.tensor([1]),
            "EOS": torch.tensor([2]),
        }

        self.modal_embedding = nn.ModuleDict()
        for modal in self.config.modal_list:
            self.modal_embedding[modal] = nn.Linear(
                self.config[f"{modal}_dim"], self.config.d_model
            )

        # self.enc_layers = StackedAttention(
        #     d_model=self.config.d_model,
        #     d_value=self.config.d_model,
        #     d_hidden=self.config.d_hidden,
        #     nhead=8,
        #     dropout=0.1,
        #     layer=self.config.transformer_layer,
        # )

        # token: [CLS], [EOS], [SEP]
        self.token_embedding = nn.Embedding(3, self.config.d_model)

        self.enc_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.config.d_model, nhead=8,),
            num_layers=self.config.transformer_layer,
        )

        # self.attn_pooling_classifier = nn.ModuleList()
        # for _ in range(self.config.num_classes):
        #     self.attn_pooling_classifier.append(
        #         AttentionPoolingClassifier(
        #             embed_dim=self.config.d_model, dropout=self.config.dropout
        #         )
        #     )
        self.classifier = nn.ModuleList()
        for _ in range(self.config.num_classes):
            self.classifier.append(
                nn.Sequential(
                    nn.Dropout(self.config.dropout), nn.Linear(self.config.d_model, 1),
                )
            )

    def forward(self, inputs):
        feature_dict = dict()
        mask_dict = dict()

        for modal in self.config.modal_list:
            feature_dict[modal] = self.modal_embedding[modal](
                inputs[f"{modal}_feature"].squeeze().type(torch.float)
            )
            feature_dict[modal] = torch.transpose(
                feature_dict[modal], 0, 1
            )  # (sq_len, batch_size, d_model)
            mask_dict[modal] = inputs[f"{modal}_mask"]

        # feature = torch.cat(list(feature_dict.values()), dim=0)
        # mask = torch.cat(list(mask_dict.values()), dim=-1)

        assert len(feature_dict) == len(mask_dict)

        feature = list()
        mask = list()

        # add CLS
        cls_embed = self.token_embedding(self.token_dict[CLS].to(self.device)).squeeze()
        sep_embed = self.token_embedding(self.token_dict[SEP].to(self.device)).squeeze()
        eos_embed = self.token_embedding(self.token_dict[EOS].to(self.device)).squeeze()

        for modal in feature_dict:
            temp_feature = feature_dict[modal]
            temp_mask = mask_dict[modal]

            batch_size = temp_feature.size(1)
            sep_ = torch.stack([sep_embed] * batch_size).unsqueeze(0)
            temp_feature = torch.cat([temp_feature, sep_], dim=0)
            temp_mask = torch.cat(
                [torch.zeros((batch_size, 1)).type_as(temp_mask), temp_mask], dim=-1
            )

            feature.append(temp_feature)
            mask.append(temp_mask)

        cls_ = torch.stack([cls_embed] * batch_size).unsqueeze(0)
        eos_ = torch.stack([eos_embed] * batch_size).unsqueeze(0)

        feature = torch.cat([cls_, *feature, eos_], dim=0)
        mask = torch.cat(
            [
                torch.zeros((batch_size, 1)).type_as(mask[0]),
                *mask,
                torch.zeros((batch_size, 1)).type_as(mask[0]),
            ],
            dim=-1,
        )

        feature = self.enc_layers(feature, src_key_padding_mask=mask.type(torch.bool))

        result = list()
        for idx in range(self.config.num_classes):
            # temp_result = self.attn_pooling_classifier[idx](feature, None)
            temp_result = self.classifier[idx](feature[0, :, :])
            result.append(temp_result)

        result = torch.cat(result, dim=-1)
        result = nn.Sigmoid()(result)

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
