import torch
import torch.nn as nn


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
