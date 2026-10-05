import torch

from mb_explainer.arithmetic import (
    arithmetic_intensity,
    dtype_bytes,
    flops_per_byte,
    gemm_bytes,
    gemm_flops,
    ridge_intensity,
    ridge_tokens,
)
from mb_explainer.catalog import ROOFLINE_MODEL_ID, head_dim, qwen_by_id
from mb_explainer.machine import float32_peak, triad
from mb_explainer.models import qwen_layer_extras, qwen_projection_elements, qwen_projection_shape
from mb_explainer.timing import fastest, median, samples


def matrix_spec() -> dict:
    row = qwen_by_id(ROOFLINE_MODEL_ID)
    dim = head_dim(row)
    inner, cols = qwen_projection_shape(row["hidden"], row["intermediate"], row["kvHeads"], dim)
    extras = qwen_layer_extras(row["hidden"], row["kvHeads"], dim)
    elements = qwen_projection_elements(row["hidden"], row["intermediate"], row["kvHeads"], dim)
    return {
        "model": row["id"],
        "revision": row["revision"],
        "configSource": row["configSource"],
        "description": (
            "Projections from one transformer layer, stacked as one matrix with the "
            "same weight count and the same multiply-adds. RMSNorm weights and q, k, v "
            "biases are not in the matrix."
        ),
        "rows": inner,
        "cols": cols,
        "dtype": "float32",
        "bytesPerElement": dtype_bytes("float32"),
        "weightElements": elements,
        "excludedBiasElements": extras["bias"],
        "excludedNormElements": extras["norm"],
    }


def per_weight() -> list[dict]:
    rows = []
    for name in ("float32", "float16", "bfloat16", "int8"):
        width = dtype_bytes(name)
        rows.append(
            {
                "dtype": name,
                "bytesPerWeight": width,
                "flopsPerWeight": 2,
                "flopsPerByteAtBatch1": flops_per_byte(1, width),
            }
        )
    return rows


def measure_point(weight: torch.Tensor, tokens: int, repeats: int) -> dict:
    inner, cols = weight.shape
    left = torch.randn(tokens, inner, dtype=weight.dtype)
    out = torch.empty(tokens, cols, dtype=weight.dtype)
    took = samples(lambda: torch.mm(left, weight, out=out), repeats, 1)
    out[0, 0].item()
    flops = gemm_flops(tokens, inner, cols)
    nbytes = gemm_bytes(tokens, inner, cols, weight.element_size())
    mid = median(took)
    return {
        "tokens": tokens,
        "flops": flops,
        "bytes": nbytes,
        "arithmeticIntensity": arithmetic_intensity(flops, nbytes),
        "seconds": took,
        "medianSeconds": mid,
        "bestSeconds": fastest(took),
        "achievedFlopsPerSecond": flops / mid,
        "trafficBytesPerSecond": nbytes / mid,
    }


def ceilings(machine: dict, spec: dict) -> dict:
    bandwidth = triad(machine)["bestBytesPerSecond"]
    peak = float32_peak(machine)["bestFlopsPerSecond"]
    ridge = ridge_intensity(peak, bandwidth)
    return {
        "bandwidthBytesPerSecond": bandwidth,
        "bandwidthKernel": "float64 triad, best of the timed calls",
        "peakFlopsPerSecond": peak,
        "peakKernel": "float32 square gemm, best of the timed calls",
        "ridgeArithmeticIntensity": ridge,
        "analyticalRidgeTokens": ridge_tokens(
            spec["rows"], spec["cols"], spec["bytesPerElement"], ridge
        ),
    }


def measured_ridge(points: list[dict], peak: float) -> int | None:
    for point in points:
        if point["achievedFlopsPerSecond"] >= 0.8 * peak:
            return int(point["tokens"])
    return None


def sweep(machine: dict, tokens: tuple[int, ...], repeats: int) -> dict:
    spec = matrix_spec()
    weight = torch.randn(spec["rows"], spec["cols"], dtype=torch.float32)
    points = [measure_point(weight, count, repeats) for count in tokens]
    caps = ceilings(machine, spec)
    caps["measuredRidgeTokens"] = measured_ridge(points, caps["peakFlopsPerSecond"])
    caps["measuredRidgeFractionOfPeak"] = 0.8
    return {"matrix": spec, "perWeight": per_weight(), "ceilings": caps, "points": points}
