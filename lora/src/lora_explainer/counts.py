def full_params(d: int, k: int) -> int:
    return d * k


def lora_params(d: int, k: int, r: int) -> int:
    return r * (d + k)


def adapter_params(d: int, layers: int, matrices: int, r: int) -> int:
    return layers * matrices * lora_params(d, d, r)


def update_multiply_adds(d: int, k: int, r: int) -> int:
    return r * (d + k)


def full_multiply_adds(d: int, k: int) -> int:
    return d * k


def unmerged_layer_multiply_adds(d: int, k: int, r: int) -> int:
    return full_multiply_adds(d, k) + update_multiply_adds(d, k, r)


def adapter_bytes(params: int, bytes_per_param: int) -> int:
    return params * bytes_per_param
