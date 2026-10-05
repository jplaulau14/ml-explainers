def gb(value: float) -> str:
    return f"{value / 1e9:.3f}"


def gflops(value: float) -> str:
    return f"{value / 1e9:.3f}"


def rate(value: float) -> str:
    return f"{value:.3f}"


def ratio(value: float) -> str:
    return f"{value:.3f}"
