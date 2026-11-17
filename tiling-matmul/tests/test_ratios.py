import pytest

from tiling_explainer.ratios import gflops, ratios_to, summarize

RUNS = [
    {"naive": 200.0, "tile32": 100.0},
    {"naive": 300.0, "tile32": 250.0},
    {"naive": 100.0, "tile32": 40.0},
]


def test_ratio_is_naive_time_over_kernel_time() -> None:
    assert ratios_to("naive", RUNS[0]) == {"naive": 1.0, "tile32": 2.0}


def test_summary_range_is_the_per_run_extremes() -> None:
    row = summarize("tile32", 4096, RUNS)
    assert row["ratioMin"] == pytest.approx(1.2)
    assert row["ratioMax"] == pytest.approx(2.5)


def test_gflops_this_run_is_the_median_run() -> None:
    row = summarize("tile32", 4096, RUNS)
    assert row["gflopsThisRun"] == pytest.approx(gflops(4096, 100.0))
    assert gflops(4096, 100.0) == pytest.approx(2 * 4096**3 / 0.1 / 1e9)
