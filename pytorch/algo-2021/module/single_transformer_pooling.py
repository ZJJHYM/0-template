import math

import torch
import torch.nn as nn

from module.base import VideoClassifier


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
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

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)

        self.activation = nn.ReLU()

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
                q = layer(q, q, q_mask, None)
            else:
                q = layer(q, v, v_mask, None)

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


class Mixer(nn.Module):
    def __init__(self, patch, channel, hidden_dim, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(patch)
        self.patch_mlp = nn.Sequential(
            nn.Linear(patch, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, patch)
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(channel)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channel, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, channel)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.patch_mlp.apply(weights_init)
        self.channel_mlp.apply(weights_init)

    def forward(self, inputs):
        # inputs -> (batch_size, patch, channel)
        x = torch.transpose(inputs, -2, -1)
        x2 = self.norm1(x)
        x2 = self.patch_mlp(x2)
        x = self.dropout1(x2) + x

        x = torch.transpose(x, -2, -1)
        x2 = self.norm2(x)
        x2 = self.channel_mlp(x2)
        x = self.dropout2(x2) + x

        return x


class StackedMixer(nn.Module):
    def __init__(self, patch, channel, hidden_dim, dropout, layer):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(layer):
            self.layers.append(Mixer(patch, channel, hidden_dim, dropout))

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x


class ContextGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ctx_gate_w = nn.Linear(in_features=dim, out_features=dim)

        nn.init.xavier_normal_(self.ctx_gate_w.weight)

    def forward(self, inputs):
        gated = self.ctx_gate_w(inputs)
        gated = nn.Sigmoid()(gated)

        return torch.mul(gated, inputs)


class CausalNormClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        feat_dim,
        use_effect=True,
        num_heads=2,
        tau=16.0,
        alpha=0.5,
        gamma=0.03125,
    ):
        super().__init__()

        self.weight = nn.parameter.Parameter(
            torch.rand(num_classes, feat_dim), requires_grad=True
        )
        self.scale = tau / num_heads
        self.norm_scale = gamma
        self.alpha = alpha
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.use_effect = use_effect
        self._reset_parameters(self.weight)
        self.relu = nn.ReLU(inplace=True)

    def _reset_parameters(self, weight):
        stdv = 1.0 / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def get_cos_sin(self, x, y):
        cos_val = (
            (x * y).sum(-1, keepdim=True)
            / torch.norm(x, 2, 1, keepdim=True)
            / torch.norm(y, 2, 1, keepdim=True)
        )
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_heads
        assert len(y_list) == self.num_heads
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def capsule_norm(self, x):
        norm = torch.norm(x.clone(), 2, 1, keepdim=True)
        normed_x = (norm / (1 + norm)) * (x / norm)
        return normed_x

    def causal_norm(self, x, weight):
        norm = torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x

    def forward(self, x, embed):
        # calculate capsule normalized feature vector and predict
        normed_w = self.multi_head_call(
            self.causal_norm, self.weight, weight=self.norm_scale
        )
        normed_x = self.multi_head_call(self.l2_norm, x)
        y = torch.mm(normed_x * self.scale, normed_w.t())

        # remove the effect of confounder c during test
        if (not self.training) and self.use_effect:
            self.embed = embed.view(1, -1)
            normed_c = self.multi_head_call(self.l2_norm, self.embed)
            head_dim = x.shape[1] // self.num_heads
            x_list = torch.split(normed_x.type_as(x), head_dim, dim=1)
            c_list = torch.split(normed_c.type_as(x), head_dim, dim=1)
            w_list = torch.split(normed_w.type_as(x), head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, _ = self.get_cos_sin(nx, nc)
                y0 = torch.mm((nx - cos_val * self.alpha * nc) * self.scale, nw.t())
                output.append(y0)
            y = sum(output)

        return y


class SingleTransPoolingClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.embed_mean = torch.zeros(self.config.d_model)
        self.mu = 0.99

        self.cls = nn.parameter.Parameter(
            torch.rand(1, self.config.d_model), requires_grad=True,
        )
        self.pooling = StackedAttention(
            d_model=self.config.d_model,
            d_hidden=self.config.hidden_dim,
            nhead=self.config.pooling_head,
            dropout=0.1,
            layer=self.config.pooling_layer,
        )

        # classifiers
        self.classifier = CausalNormClassifier(
            num_classes=self.config.num_classes, feat_dim=self.config.d_model
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.3),
        #     nn.Linear(self.config.d_model, self.config.hidden_dim),
        #     ContextGating(self.config.hidden_dim),
        #     nn.Linear(self.config.hidden_dim, self.config.num_classes),
        #     nn.Sigmoid(),
        # )

        # self.classifier.apply(weights_init)
        # classifiers = list()
        # for _ in range(self.config["num_classes"]):
        #     classifiers.append(
        #         nn.Sequential(
        #             # nn.Dropout(self.config.dropout),
        #             nn.Linear(self.config.d_model, self.config.hidden_dim),
        #             nn.ReLU(),
        #             nn.Linear(self.config.hidden_dim, 1),
        #             # nn.Linear(self.config.d_model, self.config.pooling_dim),
        #             # SE(
        #             #     embed_dim=self.config.pooling_dim,
        #             #     hidden_dim=self.config.pooling_dim,
        #             #     dropout=self.config.dropout,
        #             # ),
        #             # nn.Linear(self.config.pooling_dim, 1),
        #             nn.Sigmoid(),
        #         )
        #     )

        # self.classifiers = nn.ModuleList(classifiers)
        # for classifier in self.classifiers:
        #     classifier.apply(weights_init)

    def forward(self, inputs):
        feature_dict = dict()
        mask_dict = dict()

        for modal in self.config.modal_list:
            feature_dict[modal] = inputs[f"{modal}_feature"].squeeze().type(torch.float)
            mask_dict[modal] = inputs[f"{modal}_mask"].type(torch.float)

        feature_dict["audio"] = torch.cat([feature_dict["audio"]] * 6, dim=-1)

        feature = torch.cat(list(feature_dict.values()), dim=1)
        feature = torch.transpose(feature, 0, 1)

        mask = torch.cat(list(mask_dict.values()), dim=-1)
        batch_size = feature.size(1)

        cls_token = torch.transpose(torch.stack([self.cls] * batch_size), 0, 1)
        feature = torch.cat([cls_token, feature], dim=0)

        mask = torch.cat([torch.zeros(batch_size, 1).type_as(mask), mask], dim=-1)
        feature = self.pooling(q=feature, q_mask=mask, is_self_attention=True)

        # result = list()
        # for idx, classifier in enumerate(self.classifiers):
        #     temp_feature = feature[idx, :, :]
        #     temp_result = classifier(temp_feature)
        #     result.append(temp_result)

        # result = torch.cat(result, dim=-1)
        feature = feature[0, :, :]

        if self.training:
            self.embed_mean = self.mu * self.embed_mean.type_as(feature) + feature.mean(
                0
            ).view(-1)

        result = self.classifier(feature, self.embed_mean)

        return {
            "predicted_label": result,
        }
