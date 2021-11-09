import dgl
import torch
import torch.nn as nn
from dgl.nn import GraphConv

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


class LabelGraphConv(nn.Module):
    def __init__(self, num_classes, embedding_dim, out_feats):
        super().__init__()

        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.gconv1 = GraphConv(
            in_feats=embedding_dim, out_feats=out_feats, weight=True, bias=True
        )

    #         self.gconv2 = GraphConv(
    #             in_feats=out_feats, out_feats=out_feats, weight=True, bias=True
    #         )

    def forward(self, graph):
        node_feature = self.embedding(graph.ndata["feature"].squeeze())

        edge_weight = graph.edata["w"].type_as(node_feature)
        node_feature = self.gconv1(graph, node_feature, edge_weight=edge_weight)
        #         node_feature = self.gconv2(graph, node_feature, edge_weight=edge_weight)

        return node_feature


class GCNClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        data, _ = dgl.load_graphs(self.config.label_graph)
        self.label_graph = data[0]
        # use one-hot embedding for each label
        self.label_graph.ndata["feature"] = torch.arange(
            0, self.config.num_classes, dtype=torch.int
        ).reshape(-1, 1)

        self.label_gconv = LabelGraphConv(
            num_classes=self.config.num_classes, embedding_dim=1024, out_feats=2048,
        )

        self.cls = nn.parameter.Parameter(torch.rand(1, 768), requires_grad=True)
        self.pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=self.config.pooling_head),
            num_layers=self.config.pooling_layer,
        )

        self.feature_augment = nn.Sequential(
            nn.Dropout(self.config.dropout), nn.Linear(768, 2048)
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

        feature_dict["audio"] = torch.cat([feature_dict["audio"]] * 6, dim=-1)

        feature = torch.cat(list(feature_dict.values()), dim=0)
        mask = torch.cat(list(mask_dict.values()), dim=-1)
        batch_size = feature.size(1)
        cls_token = torch.stack([self.cls] * batch_size)
        cls_token = torch.transpose(cls_token, 0, 1)

        feature = torch.cat([cls_token, feature], dim=0)
        mask = torch.cat([torch.zeros((batch_size, 1)).type_as(mask), mask], dim=-1)
        feature = self.pooling(feature, src_key_padding_mask=mask.type(torch.bool))
        feature = feature[0, :, :]
        feature = self.feature_augment(feature)

        label_feature = self.label_gconv(self.label_graph.to(self.device))

        result = torch.matmul(label_feature, torch.transpose(feature, 0, 1))
        result = torch.transpose(result, 0, 1)
        result = nn.Sigmoid()(result)

        return {
            "predicted_label": result,
        }
