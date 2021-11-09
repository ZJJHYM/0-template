import dgl
from dgl.nn import GraphConv, EdgeWeightNorm
import torch
import torch.nn as nn
from model.utils import weights_init

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


class ProjectionMixer(nn.Module):
    def __init__(
        self,
        patch,
        channel,
        projected_patch,
        projected_channel,
        hidden_dim,
        dropout,
        layer,
    ):
        super().__init__()

        self.projected_patch = projected_patch is not None
        self.projected_channel = projected_channel is not None

        self.layers = nn.ModuleList()
        for _ in range(layer):
            self.layers.append(
                Mixer(
                    patch=patch, channel=channel, hidden_dim=hidden_dim, dropout=dropout
                )
            )

        if self.projected_patch:
            self.patch_projector = nn.Sequential(
                nn.Linear(patch, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, projected_patch),
            )

        if self.projected_channel:
            self.channel_projector = nn.Sequential(
                nn.Linear(channel, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, projected_channel),
            )

    def forward(self, inputs):
        # inputs -> (patch, batch_size, channel)
        x = inputs
        x = x.permute(1, 0, 2)

        for layer in self.layers:
            x = layer(x)

        # x -> (batch_size, patch, channel)
        if self.projected_patch:
            x = self.patch_projector(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.projected_channel:
            x = self.channel_projector(x)

        return x.permute(1, 0, 2)  # -> (patch, batch_size, channel)


class AttentionPooling(nn.Module):
    def __init__(
        self, embed_dim, hidden_dim, num_heads, dropout, kdim, num_classes, layer
    ):
        super().__init__()
        self.q = nn.parameter.Parameter(torch.rand(num_classes, embed_dim))
        self.attn = StackedAttention(
            d_model=embed_dim,
            d_value=kdim,
            d_hidden=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            layer=layer,
        )

        self.layer_norm_final = nn.LayerNorm(embed_dim)

    def forward(self, feature, mask):
        # expand self.q to batch size
        batch_size = feature.shape[1]
        expanded_q = torch.stack(
            [self.q] * batch_size, dim=0
        )  # (N, num_classes, embed_dim)
        expanded_q = torch.transpose(
            expanded_q, 0, 1
        )  # (num_classes, batch_size, embed_dim)

        # (num_classes, N, embed_dim)
        pooled_inputs = self.attn(expanded_q, feature, None, mask)
        return pooled_inputs  # (num_classes, batch_size, embed_dim)


class ContextGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ctx_gate_w = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.ctx_gate_w.weight)

    def forward(self, inputs):
        gated = self.ctx_gate_w(inputs)
        gated = nn.Sigmoid()(gated)

        return torch.mul(gated, inputs)


class LabelGraphClassifier(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()

        self.gconv1 = GraphConv(
            in_feats=in_feats, out_feats=out_feats, weight=True, bias=True
        )

    #         self.gconv2 = GraphConv(
    #             in_feats=out_feats, out_feats=out_feats, weight=True, bias=True
    #         )

    def forward(self, graph):
        graph = dgl.add_self_loop(graph)

        node_feature = graph.ndata["feature"]
        edge_weight = graph.edata["w"].type_as(node_feature)
        node_feature = self.gconv1(graph, node_feature, edge_weight=edge_weight)
        #         node_feature = self.gconv2(graph, node_feature, edge_weight=edge_weight)

        return node_feature


class CoAttnClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        data, _ = dgl.load_graphs(self.config.label_graph)
        self.label_graph = data[0]
        self.label_graph.ndata["feature"] = torch.eye(self.config.num_classes)

        # modal co attention models
        # use a ModuleDict to store
        total_dim = 0
        self.co_attention_dict = nn.ModuleDict()
        for modal in self.config.modal_list:
            total_dim += self.config[f"{modal}_dim"]
            for other_modal in self.config.modal_list:
                if modal == other_modal:
                    continue

                self.co_attention_dict[f"{modal}_{other_modal}"] = StackedAttention(
                    d_model=self.config[f"{modal}_dim"],
                    d_value=self.config[f"{other_modal}_dim"],
                    d_hidden=self.config.hidden_dim,
                    nhead=self.config.attention_head,
                    dropout=self.config.attention_dropout,
                    layer=self.config.co_attention_layer,
                )

        # modal self attention models
        # use a ModuleDict to store
        self.self_attention_dict = nn.ModuleDict()
        # self.patch_projection_dict = nn.ModuleDict()
        self.pooling_dict = nn.ModuleDict()
        for modal in self.config.modal_list:
            self.self_attention_dict[modal] = StackedAttention(
                d_model=self.config[f"{modal}_dim"] * (len(self.config.modal_list) - 1),
                d_value=self.config[f"{modal}_dim"] * (len(self.config.modal_list) - 1),
                d_hidden=self.config.hidden_dim,
                nhead=self.config.attention_head,
                dropout=self.config.attention_dropout,
                layer=self.config.self_attention_layer,
            )

            # self.patch_projection_dict[modal] = ProjectionMixer(
            #     patch=self.config[f"{modal}_padding_size"],
            #     channel=self.config[f"{modal}_dim"] * (len(self.config.modal_list) - 1),
            #     projected_patch=1,
            #     projected_channel=self.config.pooling_dim,
            #     hidden_dim=self.config.hidden_dim,
            #     dropout=self.config.attention_dropout,
            #     layer=self.config.mixer_layer,
            # )

            # self.pooling_dict[modal] = StackedAttention(
            #     d_model=self.config.pooling_dim,
            #     d_value=self.config[f"{modal}_dim"] * (len(self.config.modal_list) - 1),
            #     d_hidden=self.config.hidden_dim,
            #     nhead=8,
            #     dropout=self.config.attention_dropout,
            #     layer=self.config.pooling_layer,
            # )

            self.pooling_dict[modal] = AttentionPooling(
                embed_dim=self.config.pooling_dim,
                hidden_dim=self.config.hidden_dim,
                num_classes=1,
                num_heads=8,
                dropout=0.1,
                kdim=self.config[f"{modal}_dim"] * (len(self.config.modal_list) - 1),
                layer=3,
            )

        self.before_classifier = nn.Sequential(
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(
                self.config.pooling_dim * len(self.config.modal_list),
                self.config.pooling_dim,
            ),
        )
        self.label_gcn = LabelGraphClassifier(
            in_feats=self.config.num_classes, out_feats=self.config.pooling_dim
        )

    def forward(self, inputs):
        feature_dict = dict()
        mask_dict = dict()

        for modal in self.config.modal_list:
            feature_dict[modal] = (
                inputs[f"{modal}_feature"].squeeze().type(torch.float)
            )  # (batch_size, padding_size, dim)
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
            # !! if pass q and v, the mechanism is different from self attention, for only q will be updated
            # !! while in self attention, q and v will be updated simultaneously
            feature_dict[modal] = self.self_attention_dict[modal](
                q=feature_dict[modal],
                v=feature_dict[modal],
                q_mask=mask_dict[modal],
                v_mask=mask_dict[modal],
            )

            # now pooling sequences of features with attn pooling
            # temp_query = self.patch_projection_dict[modal](feature_dict[modal])

            # feature_dict[modal] = self.pooling_dict[modal](
            #     q=temp_query,
            #     v=feature_dict[modal],
            #     q_mask=None,
            #     v_mask=mask_dict[modal].type(torch.bool),
            # )
            feature_dict[modal] = self.pooling_dict[modal](
                feature_dict[modal], mask_dict[modal].type(torch.bool)
            )

        # now concat modal features and feed into classifiers
        feature = torch.cat(list(feature_dict.values()), dim=-1).squeeze()

        feature = self.before_classifier(feature)  # (batch_size, pooling_dim)

        label_embedding = self.label_gcn(
            self.label_graph.to(self.device)
        )  # (num_classes, pooling_dim)

        result = torch.matmul(label_embedding, torch.transpose(feature, 0, 1))
        result = torch.transpose(result, 0, 1)
        result = nn.Sigmoid()(result)

        return {
            "predicted_label": result,
        }
