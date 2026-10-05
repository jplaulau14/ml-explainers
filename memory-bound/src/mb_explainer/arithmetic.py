def dtype_bytes(name: str) -> int:
    table = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}
    return table[name]


def gemm_flops(rows: int, inner: int, cols: int) -> int:
    return 2 * rows * inner * cols


def gemm_bytes(rows: int, inner: int, cols: int, bytes_per_element: int) -> int:
    terms = rows * inner + inner * cols + rows * cols
    return terms * bytes_per_element


def arithmetic_intensity(flops: float, nbytes: float) -> float:
    return flops / nbytes


def flops_per_weight(tokens: int) -> int:
    return 2 * tokens


def flops_per_byte(tokens: int, bytes_per_weight: int) -> float:
    return flops_per_weight(tokens) / bytes_per_weight


def tokens_per_second(bandwidth: float, bytes_per_token: float) -> float:
    return bandwidth / bytes_per_token


def ridge_intensity(peak_flops: float, bandwidth: float) -> float:
    return peak_flops / bandwidth


def ridge_tokens(inner: int, cols: int, bytes_per_element: int, ridge: float) -> float | None:
    numer = ridge * bytes_per_element * inner * cols
    denom = 2 * inner * cols - ridge * bytes_per_element * (inner + cols)
    if denom <= 0:
        return None
    return numer / denom


def expected_draft_tokens(acceptance: float, draft_count: int) -> float:
    if acceptance == 1:
        return float(draft_count + 1)
    return (1 - acceptance ** (draft_count + 1)) / (1 - acceptance)
