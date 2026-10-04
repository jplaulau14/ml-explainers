import torch
from transformers import GPT2LMHeadModel

from lora_explainer.adapter import BLOCK_INDEX, LoraAttention, Which, block_slice


def attention_layer(model: GPT2LMHeadModel, layer: int) -> torch.nn.Module:
    return model.transformer.h[layer].attn.c_attn


def attention_weight(model: GPT2LMHeadModel, layer: int, which: Which) -> torch.Tensor:
    c_attn = attention_layer(model, layer)
    conv = c_attn.base if isinstance(c_attn, LoraAttention) else c_attn
    width = conv.weight.shape[0]
    return conv.weight[:, block_slice(which, width)].T.detach().to(torch.float64).clone()


def snapshot(model: GPT2LMHeadModel, layer: int) -> dict[Which, torch.Tensor]:
    return {which: attention_weight(model, layer, which) for which in BLOCK_INDEX}


def full_deltas(
    before: dict[Which, torch.Tensor], after: dict[Which, torch.Tensor]
) -> dict[Which, torch.Tensor]:
    return {which: after[which] - before[which] for which in BLOCK_INDEX}


def lora_deltas(model: GPT2LMHeadModel, layer: int) -> dict[Which, torch.Tensor]:
    c_attn = attention_layer(model, layer)
    return {which: c_attn.delta(which).detach().to(torch.float64) for which in BLOCK_INDEX}
