import torch
import torch.nn as nn
import math


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


class CoAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, kdim):
        super().__init__()
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=kdim,
            vdim=kdim,
        )
        self.layer_norm_final = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs):
        q = inputs["q"]
        k = inputs["k"]
        key_padding_mask = inputs["key_padding_mask"]
        attn_mask = inputs["attn_mask"]

        q_with_attn, _ = self.attention_layer(
            q, k, k, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )

        updated_q = self.layer_norm_final(q + q_with_attn)
        updated_q = self.feed_forward(updated_q) + q + q_with_attn

        return {
            "q": updated_q,
            "k": k,
            "key_padding_mask": key_padding_mask,
            "attn_mask": attn_mask,
        }


class StackedCoAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, kdim, layer=1):
        super().__init__()
        modules = list()
        for _ in range(layer):
            modules.append(
                CoAttention(
                    embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, kdim=kdim
                )
            )

        self.num_heads = num_heads
        self.module_list = nn.ModuleList(modules)

    def forward(self, q, k, q_mask, k_mask):
        q_k_mask = torch.matmul(q_mask.unsqueeze(2), k_mask.unsqueeze(1))

        q_k_mask = torch.cat([q_k_mask] * self.num_heads, dim=0).type(torch.bool)

        for attn in self.module_list:
            result = attn(
                {
                    "q": q,
                    "k": k,
                    "key_padding_mask": k_mask.type(torch.bool),
                    "attn_mask": q_k_mask,
                }
            )

            q = result["q"]

        return q
