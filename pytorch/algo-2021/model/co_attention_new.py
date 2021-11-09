import torch
import torch.nn as nn


class CoAttention(nn.Module):
    # follow PyTorch TransformerEncoderLayer
    # reference: https://pytorch.org/docs/1.8.1/_modules/torch/nn/modules/transformer.html#TransformerEncoder
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout, kdim):
        super().__init__()
        self.co_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=kdim,
            vdim=kdim,
        )
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        q = inputs["q"]
        k = inputs["k"]
        key_padding_mask = inputs["key_padding_mask"]
        attn_mask = inputs["attn_mask"]

        q_with_attn, _ = self.co_attn(
            q, k, k, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )

        q = q + self.dropout1(q_with_attn)
        q = self.norm1(q)

        q_with_ff = self.linear2(self.dropout(self.activation(self.linear1(q))))
        q = q + self.dropout2(q_with_ff)
        q = self.norm2(q)

        return {
            "q": q,
            "k": k,
            "key_padding_mask": key_padding_mask,
            "attn_mask": attn_mask,
        }


class StackedCoAttention(nn.Module):
    def __init__(
        self, embed_dim, hidden_dim=2048, num_heads=8, dropout=0.1, kdim=None, layer=1
    ):
        super().__init__()
        modules = list()
        for _ in range(layer):
            modules.append(
                CoAttention(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    kdim=kdim,
                )
            )

        self.num_heads = num_heads
        self.module_list = nn.ModuleList(modules)

    def forward(self, q, k, q_mask, k_mask):
        if q_mask is not None and k_mask is not None:
            q_k_mask = torch.matmul(q_mask.unsqueeze(2), k_mask.unsqueeze(1))

            q_k_mask = torch.cat([q_k_mask] * self.num_heads, dim=0).type(torch.bool)
            k_mask = k_mask.type(torch.bool)
        else:
            q_k_mask = None

        for attn in self.module_list:
            result = attn(
                {"q": q, "k": k, "key_padding_mask": k_mask, "attn_mask": q_k_mask,}
            )

            q = result["q"]

        return q
