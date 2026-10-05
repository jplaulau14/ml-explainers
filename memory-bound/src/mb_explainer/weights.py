import torch


def _seen_add(seen: set[int], ptr: int, nbytes: int) -> int:
    if ptr in seen:
        return 0
    seen.add(ptr)
    return nbytes


def unique_parameter_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        total += _seen_add(seen, param.data_ptr(), param.numel() * param.element_size())
    return total


def unique_parameter_count(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        total += _seen_add(seen, param.data_ptr(), param.numel())
    return total


def linear_weight_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight
        total += _seen_add(seen, weight.data_ptr(), weight.numel() * weight.element_size())
    return total


def packed_weight_bytes(model: torch.nn.Module) -> tuple[int, int, int]:
    total = 0
    elements = 0
    linears = 0
    for module in model.modules():
        packed = getattr(module, "_packed_params", None)
        if packed is None or not hasattr(packed, "_weight_bias"):
            continue
        weight, bias = packed._weight_bias()
        raw = weight.int_repr()
        total += raw.numel() * raw.element_size()
        elements += raw.numel()
        linears += 1
        if bias is not None:
            total += bias.numel() * bias.element_size()
            elements += bias.numel()
    return total, elements, linears


def _module_bytes(model: torch.nn.Module, seen: set[int]) -> tuple[int, int]:
    total = 0
    elements = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            continue
        for param in module.parameters(recurse=False):
            ptr = param.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            width = param.numel()
            total += width * param.element_size()
            elements += width
    return total, elements


def stored_bytes(model: torch.nn.Module) -> tuple[int, int, int, int]:
    packed, elements, quantized = packed_weight_bytes(model)
    body, body_elements = _module_bytes(model, set())
    stored = body + packed
    linear = linear_weight_bytes(model) + packed
    count = body_elements + elements
    return stored, linear, count, quantized
