from dataclasses import dataclass


def loads_per_multiply_add(tile: int) -> float:
    if tile < 1:
        raise ValueError(f"tile width must be at least 1, got {tile}")
    return 2 / tile


@dataclass(frozen=True)
class RegisterStep:
    loads: int
    multiply_adds: int


def register_step(rows: int, cols: int) -> RegisterStep:
    if rows < 1 or cols < 1:
        raise ValueError(f"register block must be at least 1x1, got {rows}x{cols}")
    return RegisterStep(loads=rows + cols, multiply_adds=rows * cols)
