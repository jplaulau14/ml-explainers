import math
from typing import Literal

import torch
from torch import nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

Which = Literal["q", "v"]
BLOCK_INDEX: dict[Which, int] = {"q": 0, "v": 2}


def block_slice(which: Which, width: int) -> slice:
    start = BLOCK_INDEX[which] * width
    return slice(start, start + width)


def low_rank_pair(rank: int, width: int) -> tuple[nn.Parameter, nn.Parameter]:
    a = torch.empty(rank, width)
    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
    return nn.Parameter(a), nn.Parameter(torch.zeros(width, rank))


class LoraAttention(nn.Module):
    def __init__(self, base: Conv1D, rank: int) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        self.width = base.weight.shape[0]
        self.A_q, self.B_q = low_rank_pair(rank, self.width)
        self.A_v, self.B_v = low_rank_pair(rank, self.width)

    def factors(self, which: Which) -> tuple[nn.Parameter, nn.Parameter]:
        return (self.A_q, self.B_q) if which == "q" else (self.A_v, self.B_v)

    def update(self, x: torch.Tensor, which: Which) -> torch.Tensor:
        a, b = self.factors(which)
        return x @ a.T @ b.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.update(x, "q")
        v = self.update(x, "v")
        return self.base(x) + torch.cat([q, torch.zeros_like(q), v], dim=-1)

    def delta(self, which: Which) -> torch.Tensor:
        a, b = self.factors(which)
        return b @ a

    def merged_weight(self) -> torch.Tensor:
        weight = self.base.weight.detach().clone()
        for which in BLOCK_INDEX:
            weight[:, block_slice(which, self.width)] += self.delta(which).detach().T
        return weight


def inject(model: GPT2LMHeadModel, rank: int) -> GPT2LMHeadModel:
    model.requires_grad_(False)
    for block in model.transformer.h:
        block.attn.c_attn = LoraAttention(block.attn.c_attn, rank)
    return model
