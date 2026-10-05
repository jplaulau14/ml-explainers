def gpt2_parameters(n_layer: int, n_embd: int, n_positions: int, vocab: int, n_inner: int) -> int:
    per_layer = (
        4 * n_embd
        + n_embd * (3 * n_embd)
        + 3 * n_embd
        + n_embd * n_embd
        + n_embd
        + n_embd * n_inner
        + n_inner
        + n_inner * n_embd
        + n_embd
    )
    embeddings = vocab * n_embd + n_positions * n_embd
    return embeddings + 2 * n_embd + n_layer * per_layer


def gpt2_mask_elements(n_layer: int, n_positions: int) -> int:
    return n_layer * n_positions * n_positions


def qwen_head_dim(hidden: int, heads: int) -> int:
    if hidden % heads != 0:
        raise ValueError(f"{hidden} is not divisible by {heads}")
    return hidden // heads


def qwen_parameters(
    hidden: int,
    intermediate: int,
    layers: int,
    vocab: int,
    kv_heads: int,
    head_dim: int,
    tied: bool,
) -> int:
    kdim = kv_heads * head_dim
    projections = 2 * hidden * hidden + 2 * hidden * kdim + 3 * hidden * intermediate
    bias = hidden + 2 * kdim
    body = layers * (projections + bias + 2 * hidden) + hidden + vocab * hidden
    return body if tied else body + vocab * hidden


def qwen_projection_elements(hidden: int, intermediate: int, kv_heads: int, head_dim: int) -> int:
    kdim = kv_heads * head_dim
    return 2 * hidden * hidden + 2 * hidden * kdim + 3 * hidden * intermediate


def qwen_projection_shape(
    hidden: int, intermediate: int, kv_heads: int, head_dim: int
) -> tuple[int, int]:
    elements = qwen_projection_elements(hidden, intermediate, kv_heads, head_dim)
    return hidden, elements // hidden


def qwen_layer_extras(hidden: int, kv_heads: int, head_dim: int) -> dict[str, int]:
    kdim = kv_heads * head_dim
    return {"bias": hidden + 2 * kdim, "norm": 2 * hidden}
