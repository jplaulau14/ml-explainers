from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    quick: bool
    stream_repeats: int
    peak_n: int
    peak_repeats: int
    peak_dtypes: tuple[str, ...]
    roofline_tokens: tuple[int, ...]
    roofline_repeats: int
    prompt_tokens: int
    new_tokens: int
    decode_repeats: int
    batches: tuple[int, ...]
    precisions: tuple[str, ...]


def settings(quick: bool) -> Settings:
    tokens = tuple(2**i for i in range(11))
    if quick:
        return Settings(
            True, 3, 2048, 2, ("float32",), (1, 4, 16, 64, 256), 1, 32, 8, 1, (1, 4), ("float32",)
        )
    return Settings(
        False,
        10,
        4096,
        3,
        ("float32", "bfloat16"),
        tokens,
        3,
        128,
        32,
        2,
        (1, 2, 4, 8),
        ("bfloat16", "float32", "int8"),
    )
