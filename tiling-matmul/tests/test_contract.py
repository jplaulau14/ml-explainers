import json
from pathlib import Path

import pytest

from tiling_explainer.conditions import PowerState
from tiling_explainer.contract import (
    CHALLENGE_KERNELS,
    SPEED_KERNELS,
    bytes_problems,
    problems,
    sidecar_problems,
)
from tiling_explainer.driver import ProcessRecord, build_machine

OUT = Path(__file__).resolve().parents[1] / "out"
FILES = [("speeds.json", SPEED_KERNELS), ("challenge.json", CHALLENGE_KERNELS)]
AC = PowerState(ac_power=True, low_power_mode=False)
BATTERY = PowerState(ac_power=False, low_power_mode=False)
VERSIONS = {"commit": "0" * 40, "macos": "26.7.3", "mlx": "0.32.4", "python": "3.12.12"}


def record(scale: float, after: PowerState = AC) -> ProcessRecord:
    kernels = {"naive": 200.0, "tile32": 190.0, "reg4x4": 80.0, "reg8x8": 280.0, "vendor": 40.0}
    results = {k: {"ms": ms * scale, "maxAbsErr": 0.0} for k, ms in kernels.items()}
    return ProcessRecord(before=AC, after=after, results=results)


def document(kernels: frozenset[str], records: list[ProcessRecord]) -> dict[str, object]:
    machine = build_machine("Apple M3 Pro", 4096, records, kernels, True, VERSIONS)
    source = {"commit": "0" * 40, "code": "https://example.invalid"}
    return {"version": 1, "source": source, "machines": [machine]}


@pytest.mark.parametrize(("name", "kernels"), FILES)
def test_committed_output_matches_the_contract(name: str, kernels: frozenset[str]) -> None:
    path = OUT / name
    if not path.exists():
        pytest.skip(f"no committed out/{name} yet")
    assert problems(json.loads(path.read_text()), kernels) == []


def test_committed_bytes_matches_the_contract() -> None:
    path = OUT / "bytes.json"
    if not path.exists():
        pytest.skip("no committed out/bytes.json yet")
    assert bytes_problems(json.loads(path.read_text())) == []


def test_committed_a10_runs_record_a_vendor_kernel_without_tensor_instructions() -> None:
    path, bytes_path = OUT / "a10-runs.json", OUT / "bytes.json"
    if not path.exists():
        pytest.skip("no committed out/a10-runs.json yet")
    sidecar, bytes_doc = json.loads(path.read_text()), json.loads(bytes_path.read_text())
    assert sidecar_problems(sidecar, bytes_doc) == []


@pytest.mark.parametrize(("name", "kernels"), FILES)
def test_harness_output_matches_the_contract(name: str, kernels: frozenset[str]) -> None:
    doc = document(kernels, [record(1.0), record(1.05), record(0.97)])
    assert problems(doc, kernels) == []


def test_mac_run_with_a_battery_reading_fails_the_contract() -> None:
    doc = document(SPEED_KERNELS, [record(1.0), record(1.0, after=BATTERY), record(1.0)])
    assert problems(doc, SPEED_KERNELS) == ["machines[0]: conditions.acPower must be true"]


def test_mac_run_with_two_processes_fails_the_contract() -> None:
    doc = document(SPEED_KERNELS, [record(1.0), record(1.0)])
    assert problems(doc, SPEED_KERNELS) == ["machines[0]: needs at least 3 runs"]


def test_nested_condition_fails_the_contract() -> None:
    doc = document(SPEED_KERNELS, [record(1.0), record(1.0), record(1.0)])
    doc["machines"][0]["conditions"]["timing"] = {"samples": 10}
    assert problems(doc, SPEED_KERNELS) == [
        "machines[0]: conditions.timing must be a string, number or boolean"
    ]


def test_version_that_is_not_a_string_fails_the_contract() -> None:
    doc = document(SPEED_KERNELS, [record(1.0), record(1.0), record(1.0)])
    doc["machines"][0]["versions"]["cuda"] = 12090
    assert problems(doc, SPEED_KERNELS) == ["machines[0]: versions.cuda must be a non-empty string"]
