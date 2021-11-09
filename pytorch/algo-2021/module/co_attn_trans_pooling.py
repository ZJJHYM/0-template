import torch
import torch.nn as nn

from module.base import VideoClassifier


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class Attention(nn.Module):
    def __init__(self, d_model, d_hidden, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
        )

        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)

    def forward(self, q, v=None, key_padding_mask=None, attn_mask=None):
        v2, _ = self.attn(
            q, v, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        v = v + self.dropout(v2)
        v = self.norm1(v)
        v2 = self.linear2(self.dropout(self.activation(self.linear1(v))))
        v = v + self.dropout2(v)
        v = self.norm2(v)

        return v


class StackedAttention(nn.Module):
    def __init__(self, d_model, d_hidden, nhead, dropout, layer):
        super().__init__()

        self.layers = nn.ModuleList()
        self.nhead = nhead
        for _ in range(layer):
            self.layers.append(
                Attention(
                    d_model=d_model, d_hidden=d_hidden, nhead=nhead, dropout=dropout,
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
                q = layer(q, q, q_mask, attn_mask)
            else:
                q = layer(q, v, v_mask, attn_mask)

        return q


class SE(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        x = self.dropout(inputs)
        activation = self.linear1(x)
        activation = self.norm1(activation)

        gates = self.linear2(activation)
        gates = self.norm2(gates)

        gates = self.linear3(gates)
        gates = self.norm3(gates)
        gates = nn.Sigmoid()(gates)

        activation = torch.multiply(activation, gates)

        return activation


class ContextGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ctx_gate_w = nn.Linear(in_features=dim, out_features=dim)

        nn.init.xavier_normal_(self.ctx_gate_w.weight)

    def forward(self, inputs):
        gated = self.ctx_gate_w(inputs)
        gated = nn.Sigmoid()(gated)

        return torch.mul(gated, inputs)


class CoAttnWithTransPoolingClassifier(VideoClassifier):
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
                    d_model=self.config.d_model,
                    d_hidden=self.config.hidden_dim,
                    nhead=self.config.co_attention_head,
                    dropout=self.config.dropout,
                    layer=self.config.co_attention_layer,
                )

        # self.pooling_dict = nn.ModuleDict()
        # self.clses = nn.ParameterDict()
        # for modal in self.config.modal_list:
        #     self.pooling_dict[modal] = StackedAttention(
        #         d_model=self.config.d_model,
        #         d_hidden=self.config.hidden_dim,
        #         nhead=self.config.co_attention_head,
        #         dropout=self.config.dropout,
        #         layer=self.config.pooling_layer,
        #     )

        #     self.clses[modal] = nn.parameter.Parameter(
        #         torch.rand((self.config.num_classes, self.config.d_model)),
        #         requires_grad=True,
        #     )
        self.cls = nn.parameter.Parameter(
            torch.rand(self.config.num_classes, self.config.d_model * 2),
            requires_grad=True,
        )
        self.pooling = StackedAttention(
            d_model=self.config.d_model * 2,
            d_hidden=self.config.hidden_dim,
            nhead=self.config.pooling_head,
            dropout=self.config.dropout,
            layer=self.config.pooling_layer,
        )

        # classifiers
        classifiers = list()
        for _ in range(self.config["num_classes"]):
            classifiers.append(
                nn.Sequential(
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.d_model * 2, self.config.pooling_dim),
                    nn.LayerNorm(self.config.pooling_dim),
                    nn.ReLU(),
                    ContextGating(self.config.pooling_dim),
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

        feature_dict["audio"] = torch.cat([feature_dict["audio"]] * 6, dim=-1)

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

            # remove co attention features
            for feature_name in need_to_pop_list:
                feature_dict.pop(feature_name)

            feature_dict[modal] = torch.cat(temp_feature_list, dim=-1)
            # mask_dict[modal] = torch.cat([mask_dict[modal]] * 2, dim=-1)

            # temp_feature = torch.cat(temp_feature_list, dim=0)
            # batch_size = temp_feature.size(1)
            # temp_cls = torch.stack(
            #     [self.clses[modal]] * batch_size
            # )  # (batch_size, num_classes, dim)
            # temp_cls = torch.transpose(temp_cls, 0, 1)  # (num_classes, batch_size, dim)
            # temp_feature = torch.cat(
            #     [temp_cls, temp_feature], dim=0
            # )  # (num_classes + sequence, batch_size, dim)
            # temp_mask = torch.cat(
            #     [
            #         torch.zeros(batch_size, self.config.num_classes).type_as(
            #             mask_dict[modal]
            #         ),
            #         mask_dict[modal],
            #         mask_dict[modal],
            #     ],
            #     dim=-1,
            # )

            # feature_dict[modal] = self.pooling_dict[modal](
            #     q=temp_feature, q_mask=temp_mask, is_self_attention=True
            # )

        # now concat modal features and feed into classifiers
        feature = torch.cat(list(feature_dict.values()), dim=0)
        mask = torch.cat(list(mask_dict.values()), dim=-1)
        batch_size = feature.size(1)

        cls_token = torch.transpose(torch.stack([self.cls] * batch_size), 0, 1)
        feature = torch.cat([cls_token, feature], dim=0)

        mask = torch.cat(
            [torch.zeros(batch_size, self.config.num_classes).type_as(mask), mask],
            dim=-1,
        )
        feature = self.pooling(q=feature, q_mask=mask, is_self_attention=True)

        result = list()
        for idx, classifier in enumerate(self.classifiers):
            temp_feature = feature[idx, :, :]
            temp_result = classifier(temp_feature)
            result.append(temp_result)

        result = torch.cat(result, dim=-1)

        return {
            "predicted_label": result,
        }
