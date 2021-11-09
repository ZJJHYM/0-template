import torch
import torch.nn as nn


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
