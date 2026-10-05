def pace(measured: float, predicted: float) -> str:
    ratio = measured / predicted
    if ratio < 0.75:
        return "slower"
    if ratio > 1.25:
        return "faster"
    return "close"


def cache_fit(nbytes: int, l3_bytes: int) -> str:
    if nbytes > l3_bytes:
        return "larger"
    return "fits"
