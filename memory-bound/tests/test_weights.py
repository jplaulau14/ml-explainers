import torch

from mb_explainer.weights import stored_bytes


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(5, 4)
        self.proj = torch.nn.Linear(4, 4, bias=True)
        self.norm = torch.nn.Parameter(torch.ones(4))
        self.head = torch.nn.Linear(4, 5, bias=False)
        self.head.weight = self.embed.weight


def test_tied_embedding_is_counted_once() -> None:
    stored, linear, count, quantized = stored_bytes(Tiny())
    assert quantized == 0
    assert count == 44
    assert stored == 176
    assert linear == 144


def test_int8_ceiling_skips_the_leftover_embedding() -> None:
    model = torch.ao.quantization.quantize_dynamic(Tiny(), {torch.nn.Linear}, dtype=torch.qint8)
    stored, linear, count, quantized = stored_bytes(model)
    assert quantized == 2
    assert count == 44
    assert stored == 68
    assert linear == 52
