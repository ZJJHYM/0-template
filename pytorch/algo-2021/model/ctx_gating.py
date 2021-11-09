import torch
import torch.nn as nn
from model.utils import weights_init


class ContextGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ctx_gate_w = nn.Linear(in_features=dim, out_features=dim)

        nn.init.xavier_normal_(self.ctx_gate_w.weight)

    def forward(self, inputs):
        gated = self.ctx_gate_w(inputs)
        gated = nn.Sigmoid()(gated)

        return torch.mul(gated, inputs)
