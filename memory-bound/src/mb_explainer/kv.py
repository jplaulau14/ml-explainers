from mb_explainer.arithmetic import tokens_per_second


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per_element: int) -> int:
    return 2 * layers * kv_heads * head_dim * bytes_per_element


def traffic_bytes(weight_bytes: int, kv_per_token: int, context: int) -> int:
    return weight_bytes + kv_per_token * context


def ceiling(bandwidth: float, weight_bytes: int, kv_per_token: int, context: int) -> float:
    return tokens_per_second(bandwidth, traffic_bytes(weight_bytes, kv_per_token, context))
