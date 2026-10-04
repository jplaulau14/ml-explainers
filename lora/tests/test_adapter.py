import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from lora_explainer.adapter import LoraAttention, inject

WIDTH = 16


def tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_embd=WIDTH,
        n_layer=2,
        n_head=2,
        vocab_size=64,
        n_positions=32,
        bos_token_id=0,
        eos_token_id=0,
    )
    return GPT2LMHeadModel(config).eval()


def randomized_layer(rank: int) -> LoraAttention:
    torch.manual_seed(rank)
    layer = LoraAttention(Conv1D(3 * WIDTH, WIDTH), rank)
    with torch.no_grad():
        for factor in (layer.A_q, layer.B_q, layer.A_v, layer.B_v):
            factor.normal_()
    return layer


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_claim_1_update_rank_is_at_most_r(rank: int) -> None:
    layer = randomized_layer(rank)
    for which in ("q", "v"):
        delta = layer.delta(which).detach()
        assert delta.shape == (WIDTH, WIDTH)
        assert torch.linalg.matrix_rank(delta) == rank


def test_claim_2_injected_model_starts_identical() -> None:
    base = tiny_model()
    adapted = inject(copy.deepcopy(base), rank=4)
    tokens = torch.randint(0, 64, (3, 10))
    with torch.no_grad():
        assert torch.equal(adapted(tokens).logits, base(tokens).logits)


def test_claim_3_merged_weight_matches_adapter() -> None:
    layer = randomized_layer(rank=4)
    merged = Conv1D(3 * WIDTH, WIDTH)
    with torch.no_grad():
        merged.weight.copy_(layer.merged_weight())
        merged.bias.copy_(layer.base.bias)
        x = torch.randn(2, 5, WIDTH)
        torch.testing.assert_close(layer(x), merged(x), atol=1e-5, rtol=0)


def test_only_adapter_parameters_train() -> None:
    model = inject(tiny_model(), rank=4)
    trainable = {name.rsplit(".", 1)[-1] for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"A_q", "B_q", "A_v", "B_v"}
    assert sum(p.requires_grad for p in model.parameters()) == 2 * 4
