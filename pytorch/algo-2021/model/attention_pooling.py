import torch
import torch.nn as nn


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
