import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tiling_explainer import driver
from tiling_explainer.a10g.gather import TooManyRefusals, gather
from tiling_explainer.a10g.ncu import (
    DRAM_READ,
    L2_BYTES,
    METRICS,
    REQUESTS,
    SHARED_LOADS,
    ProfileError,
    measured_launches,
    parse_launches,
)
from tiling_explainer.a10g.outputs import build_a10g_machine, run_problem, write_a10g
from tiling_explainer.a10g.run import profile_problem
from tiling_explainer.a10g.spec import guarded
from tiling_explainer.contract import SPEED_KERNELS, bytes_problems, problems

COMMIT = "a" * 40
CUBLAS = "void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)"
TENSOR_OP = "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128>(T1::Params)"
HEADER = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
NAIVE_TIMES = (100.0, 90.0, 120.0)
TILE32_TIMES = (80.0, 60.0, 100.0)


def smi(name: str) -> str:
    return f"{name}, 620.70.06, 1710, 300.00, GPU-1234\n"


def call(index: int, gpu_name: str = "NVIDIA A10G") -> dict[str, object]:
    times = {"naive": NAIVE_TIMES[index], "tile32": TILE32_TIMES[index]}
    times |= {"reg4x4": 18.0, "reg8x8": 14.0, "vendor": 9.0}
    gpu = {"name": gpu_name, "driver": "620.70.06", "max_sm_clock_mhz": 1710}
    gpu |= {"power_limit_w": 300.0, "uuid": f"GPU-{index}", "smCount": 80, "l2Bytes": 6291456}
    software = {"cudaRuntime": "12.9", "cudaDriver": "13.0", "cupy": "13.6.0"}
    software |= {"numpy": "2.5.5", "python": "3.12.12", "modal": "1.6.4", "cupyTf32": "0"}
    return {
        "commit": COMMIT,
        "container": f"ta-{index}",
        "gpuUuid": f"GPU-{index}",
        "gpu": gpu,
        "software": software,
        "kernels": {k: {"ms": ms, "samplesMs": [ms], "maxAbsErr": 0.0} for k, ms in times.items()},
        "attributes": {},
    }


def profile(vendor: str = CUBLAS, launches: int = 2) -> dict[str, object]:
    lines = [HEADER]
    names = ["naive", "tile32", "reg4x4", vendor]
    for index, name in enumerate(name for name in names for _ in range(launches)):
        lines += [f'"{index}","{name}","{metric}","","{1000 + index}"' for metric in METRICS]
    return {**call(0), "ncu": "Version 2025.2.0.0", "command": "ncu", "csv": "\n".join(lines)}


@pytest.mark.parametrize("name", ["NVIDIA A10", "NVIDIA L4", "NVIDIA A100-SXM4-40GB"])
def test_chip_guard_stops_any_gpu_but_the_a10g_before_timing(name: str) -> None:
    timed: list[str] = []
    result = guarded(smi(name), lambda reading: {"timed": timed.append(reading.name)})
    assert result["refused"] == f"GPU is '{name}', not 'NVIDIA A10G'; nothing was timed"
    assert timed == []


def test_chip_guard_lets_the_a10g_through() -> None:
    assert guarded(smi("NVIDIA A10G"), lambda reading: {"uuid": reading.uuid}) == {
        "uuid": "GPU-1234"
    }


class Done:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result

    def get(self) -> dict[str, object]:
        return self.result


def chips(names: list[str]) -> Callable[[], Done]:
    queue = iter(names)

    def start() -> Done:
        name = next(queue)
        return Done({"refused": name} if name != "NVIDIA A10G" else {"gpu": name})

    return start


def test_refused_calls_are_replaced_and_recorded() -> None:
    starters = {
        "timing": chips(["NVIDIA A10", "NVIDIA A10G", "NVIDIA A10G", "NVIDIA A10G"]),
        "profile": chips(["NVIDIA A10", "NVIDIA A10G"]),
    }
    accepted, refusals = gather(starters, {"timing": 3, "profile": 1})
    assert [len(accepted["timing"]), len(accepted["profile"])] == [3, 1]
    assert [r["kind"] for r in refusals] == ["timing", "profile"]


def test_seventh_refusal_fails_the_run() -> None:
    starters = {"timing": chips(["NVIDIA A10"] * 10), "profile": chips(["NVIDIA A10G"])}
    with pytest.raises(TooManyRefusals, match="7 calls refused, limit 6"):
        gather(starters, {"timing": 3, "profile": 1})


def test_six_refusals_still_finish_the_run() -> None:
    names = ["NVIDIA A10"] * 6 + ["NVIDIA A10G"] * 3
    accepted, refusals = gather({"timing": chips(names)}, {"timing": 3})
    assert (len(accepted["timing"]), len(refusals)) == (3, 6)


def test_profile_keeps_the_second_launch_of_each_kernel() -> None:
    measured = measured_launches(parse_launches(str(profile()["csv"])))
    assert {kernel: launch[L2_BYTES] for kernel, launch in measured.items()} == {
        "naive": 1001.0,
        "tile32": 1003.0,
        "reg4x4": 1005.0,
        "vendor": 1007.0,
    }
    assert measured["vendor"]["name"] == CUBLAS


def test_profile_missing_a_launch_is_refused() -> None:
    with pytest.raises(ProfileError, match="expected 2 profiled launches per kernel"):
        measured_launches(parse_launches(str(profile(launches=1)["csv"])))


def test_a10g_ratio_range_is_the_per_call_extremes() -> None:
    machine = build_a10g_machine(
        4096, [call(0), call(1), call(2)], profile(), SPEED_KERNELS, True, COMMIT
    )
    tile32 = next(row for row in machine["kernels"] if row["kernel"] == "tile32")
    assert (tile32["ratioMin"], tile32["ratioMax"]) == pytest.approx((1.2, 1.5))
    assert machine["conditions"]["distinctGpus"] == 3
    assert (
        problems(
            {"version": 1, "source": {"commit": COMMIT, "code": ""}, "machines": [machine]},
            SPEED_KERNELS,
        )
        == []
    )


def test_a10g_entry_with_another_chip_name_fails_the_contract() -> None:
    calls = [call(0, "NVIDIA A10"), call(1, "NVIDIA A10"), call(2, "NVIDIA A10")]
    machine = build_a10g_machine(4096, calls, profile(), SPEED_KERNELS, True, COMMIT)
    doc = {"version": 1, "source": {"commit": COMMIT, "code": ""}, "machines": [machine]}
    assert problems(doc, SPEED_KERNELS) == ["machines[0]: name must be 'NVIDIA A10G'"]


def test_calls_on_different_hardware_are_a_run_problem() -> None:
    other = call(2)
    other["gpu"] = {**other["gpu"], "power_limit_w": 150.0}
    assert (
        run_problem([call(0), call(1), other], profile(), COMMIT)
        == "calls ran with different gpu facts"
    )


def write_run(monkeypatch: pytest.MonkeyPatch, folder: Path, vendor: str) -> None:
    monkeypatch.setattr(driver, "ROOT", folder)
    launches = parse_launches(str(profile(vendor)["csv"]))
    calls = [call(0), call(1), call(2)]
    write_a10g(folder, 4096, calls, profile(vendor), [], launches, COMMIT, True)


def test_a10g_run_keeps_the_mac_entry_and_writes_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mac = {"id": "m3-pro", "versions": {"commit": "b" * 40}}
    (tmp_path / "speeds.json").write_text(json.dumps({"machines": [mac]}))
    write_run(monkeypatch, tmp_path, CUBLAS)
    speeds = json.loads((tmp_path / "speeds.json").read_text())
    assert [(m["id"], m["versions"]["commit"]) for m in speeds["machines"]] == [
        ("m3-pro", "b" * 40),
        ("a10g", COMMIT),
    ]
    doc = json.loads((tmp_path / "bytes.json").read_text())
    assert bytes_problems(doc) == []
    naive = doc["rows"][0]
    assert naive == {
        "kernel": "naive",
        "globalLoadRequests": 1001,
        "sharedLoads": 1001,
        "l2Bytes": 1001,
        "dramReadBytes": 1001,
    }
    assert {REQUESTS, SHARED_LOADS, L2_BYTES, DRAM_READ} <= set(METRICS)


def test_tensor_op_vendor_kernel_fails_the_bytes_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_run(monkeypatch, tmp_path, TENSOR_OP)
    doc = json.loads((tmp_path / "bytes.json").read_text())
    assert bytes_problems(doc) == [f"vendor kernel {TENSOR_OP!r} is not a SIMT sgemm"]


def test_tensor_op_vendor_kernel_stops_the_run_before_out() -> None:
    launches = parse_launches(str(profile(TENSOR_OP)["csv"]))
    assert profile_problem(launches) == f"vendor kernel {TENSOR_OP!r} is not a SIMT sgemm"
