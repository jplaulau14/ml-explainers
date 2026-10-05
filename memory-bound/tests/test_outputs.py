import json
import subprocess

import pytest

from mb_explainer.arithmetic import (
    arithmetic_intensity,
    gemm_bytes,
    gemm_flops,
    ridge_intensity,
    tokens_per_second,
)
from mb_explainer.format import gb
from mb_explainer.kv import kv_bytes_per_token
from mb_explainer.paths import OUT, REPO
from mb_explainer.settings import settings

FILES = ("machine.json", "roofline.json", "decode.json", "kv_cache.json", "specs.json")


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def rev_parse(spec: str) -> str:
    return subprocess.check_output(["git", "rev-parse", spec], cwd=REPO, text=True).strip()


def parent_list(spec: str) -> list[str]:
    text = subprocess.check_output(["git", "rev-parse", f"{spec}^@"], cwd=REPO, text=True)
    return [line for line in text.splitlines() if line]


def accepted_commits(head: str, head_parents: list[str], tip_parent: str | None) -> set[str]:
    if len(head_parents) > 1:
        found = {head, head_parents[1]}
        if tip_parent is not None:
            found.add(tip_parent)
        return found
    return {head, *head_parents}


def history() -> set[str]:
    head = rev_parse("HEAD")
    head_parents = parent_list(head)
    tip_parent = None
    if len(head_parents) > 1:
        tip_parents = parent_list(head_parents[1])
        tip_parent = tip_parents[0] if tip_parents else None
    return accepted_commits(head, head_parents, tip_parent)


def float64_triad(machine: dict) -> dict:
    return next(
        row for row in machine["streams"] if row["kernel"] == "triad" and row["dtype"] == "float64"
    )


def float32_peak(machine: dict) -> dict:
    return next(
        row for row in machine["peaks"] if row["dtype"] == "float32" and row["status"] == "ok"
    )


def test_linear_checkout_accepts_head_and_its_parent() -> None:
    assert accepted_commits("data", ["measured"], None) == {"data", "measured"}


def test_merge_checkout_uses_the_branch_tip() -> None:
    accepted = accepted_commits("merge", ["base", "tip"], "measured")
    assert accepted == {"merge", "tip", "measured"}
    assert "base" not in accepted


def test_committed_files_are_a_full_run() -> None:
    commits = history()
    for name in FILES:
        payload = load(name)
        assert payload["version"] == 1
        assert payload["quick"] is False
        assert payload["gitCommit"] in commits


def test_roofline_counts_match_the_gemm() -> None:
    roof = load("roofline.json")
    machine = load("machine.json")
    spec = roof["matrix"]
    assert [point["tokens"] for point in roof["points"]] == list(settings(False).roofline_tokens)
    for point in roof["points"]:
        flops = gemm_flops(point["tokens"], spec["rows"], spec["cols"])
        nbytes = gemm_bytes(point["tokens"], spec["rows"], spec["cols"], spec["bytesPerElement"])
        assert point["flops"] == flops
        assert point["bytes"] == nbytes
        assert point["arithmeticIntensity"] == pytest.approx(arithmetic_intensity(flops, nbytes))
        assert point["achievedFlopsPerSecond"] == pytest.approx(flops / point["medianSeconds"])
    triad = float64_triad(machine)
    peak = float32_peak(machine)
    caps = roof["ceilings"]
    assert caps["bandwidthBytesPerSecond"] == triad["bestBytesPerSecond"]
    assert caps["peakFlopsPerSecond"] == peak["bestFlopsPerSecond"]
    assert caps["ridgeArithmeticIntensity"] == pytest.approx(
        ridge_intensity(peak["bestFlopsPerSecond"], triad["bestBytesPerSecond"])
    )


def test_decode_predictions_use_measured_bandwidth() -> None:
    decode = load("decode.json")
    bandwidth = decode["bandwidthBytesPerSecond"]
    assert bandwidth == float64_triad(load("machine.json"))["bestBytesPerSecond"]
    assert [row["dtype"] for row in decode["precisions"]] == list(settings(False).precisions)
    model = decode["model"]
    for row in decode["precisions"]:
        if row["status"] != "ok":
            assert row["error"]
            continue
        assert row["predictedTokensPerSecondWeights"] == pytest.approx(
            tokens_per_second(bandwidth, row["uniqueBytes"])
        )
        combined = row["uniqueBytes"] + row["measuredKvBytesPerToken"] * row["meanContext"]
        assert row["predictedTokensPerSecondWeightsAndKv"] == pytest.approx(
            tokens_per_second(bandwidth, combined)
        )
        formula = kv_bytes_per_token(
            model["layers"], model["kvHeads"], model["headDim"], row["kvElementBytes"]
        )
        assert row["formulaKvBytesPerToken"] == formula
        assert row["measuredKvBytesPerToken"] == formula
        assert row["decodeTokensPerSecond"] == pytest.approx(
            decode["newTokens"] / row["decodeSecondsMedian"]
        )


def test_batch_totals_and_kv_ceilings() -> None:
    decode = load("decode.json")
    ok = [row["dtype"] for row in decode["precisions"] if row["status"] == "ok"]
    for dtype in ok:
        rows = [row for row in decode["batch"] if row["dtype"] == dtype]
        assert [row["batchSize"] for row in rows] == list(settings(False).batches)
        for row in rows:
            total = row["batchSize"] * decode["newTokens"] / row["decodeSecondsMedian"]
            assert row["totalTokensPerSecond"] == pytest.approx(total)
    kv = load("kv_cache.json")
    assert kv["bandwidthBytesPerSecond"] == decode["bandwidthBytesPerSecond"]
    for model in kv["models"]:
        for block in model["dtypes"]:
            per_token = kv_bytes_per_token(
                model["layers"], model["kvHeads"], model["headDim"], block["bytesPerElement"]
            )
            assert block["kvBytesPerToken"] == per_token
            for row in block["contexts"]:
                traffic = block["weightBytes"] + per_token * row["context"]
                assert row["bytesPerToken"] == traffic
                assert row["tokensPerSecondCeiling"] == pytest.approx(
                    tokens_per_second(kv["bandwidthBytesPerSecond"], traffic)
                )


def test_report_quotes_the_bandwidth() -> None:
    machine = load("machine.json")
    report = (OUT / "report.md").read_text()
    assert f"{gb(float64_triad(machine)['bestBytesPerSecond'])} GB/s" in report
    assert machine["gitCommit"] in report
