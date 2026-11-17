import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tiling_explainer import driver
from tiling_explainer.a10.gather import TooManyRefusals, gather
from tiling_explainer.a10.ncu import (
    DRAM_READ,
    L2_BYTES,
    METRICS,
    REQUESTS,
    SHARED_LOADS,
    ProfileError,
    measured_launches,
    parse_launches,
    tensor_metric,
)
from tiling_explainer.a10.outputs import build_a10_machine, run_problem, write_a10
from tiling_explainer.a10.run import profile_problem
from tiling_explainer.a10.spec import guarded
from tiling_explainer.contract import SPEED_KERNELS, bytes_problems, problems, sidecar_problems

COMMIT = "a" * 40
CUBLAS = "ampere_sgemm_128x64_nn"
TENSOR_OP = "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128>(T1::Params)"
TENSOR = "sm__inst_executed_pipe_tensor_op_hmma.sum"
HEADER = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
NAIVE_TIMES = (100.0, 90.0, 120.0)
TILE32_TIMES = (80.0, 60.0, 100.0)


def smi(name: str) -> str:
    return f"{name}, 620.70.06, 1710, 300.00, GPU-1234\n"


def call(index: int, gpu_name: str = "NVIDIA A10") -> dict[str, object]:
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


def profile(vendor: str = CUBLAS, launches: int = 2, tensor: int = 0) -> dict[str, object]:
    lines = [HEADER]
    names = ["naive", "tile32", "reg4x4", vendor]
    for index, name in enumerate(name for name in names for _ in range(launches)):
        lines += [f'"{index}","{name}","{metric}","","{1000 + index}"' for metric in METRICS]
        lines.append(f'"{index}","{name}","{TENSOR}","inst","{tensor}"')
    detail = {"ncu": "Version 2025.2.0.0", "command": "ncu", "tensorMetric": TENSOR}
    return {**call(0), **detail, "csv": "\n".join(lines)}


@pytest.mark.parametrize("name", ["NVIDIA A10G", "NVIDIA L4", "NVIDIA A100-SXM4-40GB"])
def test_chip_guard_stops_any_gpu_but_the_a10_before_timing(name: str) -> None:
    timed: list[str] = []
    result = guarded(smi(name), lambda reading: {"timed": timed.append(reading.name)})
    assert result["refused"] == f"GPU is '{name}', not 'NVIDIA A10'; nothing was timed"
    assert timed == []


def test_chip_guard_lets_the_a10_through() -> None:
    assert guarded(smi("NVIDIA A10"), lambda reading: {"uuid": reading.uuid}) == {
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
        return Done({"refused": name} if name != "NVIDIA A10" else {"gpu": name})

    return start


def test_refused_calls_are_replaced_and_recorded() -> None:
    starters = {
        "timing": chips(["NVIDIA A10G", "NVIDIA A10", "NVIDIA A10", "NVIDIA A10"]),
        "profile": chips(["NVIDIA A10G", "NVIDIA A10"]),
    }
    accepted, refusals = gather(starters, {"timing": 3, "profile": 1})
    assert [len(accepted["timing"]), len(accepted["profile"])] == [3, 1]
    assert [r["kind"] for r in refusals] == ["timing", "profile"]


def test_seventh_refusal_fails_the_run() -> None:
    starters = {"timing": chips(["NVIDIA A10G"] * 10), "profile": chips(["NVIDIA A10"])}
    with pytest.raises(TooManyRefusals, match="7 calls refused, limit 6"):
        gather(starters, {"timing": 3, "profile": 1})


def test_six_refusals_still_finish_the_run() -> None:
    names = ["NVIDIA A10G"] * 6 + ["NVIDIA A10"] * 3
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


def test_tensor_metric_is_the_first_listed_candidate_the_device_has() -> None:
    query = "Device NVIDIA A10 (GA102)\nsm__pipe_tensor_op_hmma_cycles_active  Counter  cycles\n"
    query += "sm__inst_executed_pipe_tensor_op_hmma  Counter  warp instructions\n"
    assert tensor_metric(query) == TENSOR


def test_device_without_a_tensor_metric_fails_the_profile() -> None:
    with pytest.raises(ProfileError, match="tensor metrics: \\[\\]"):
        tensor_metric("Device NVIDIA A10 (GA102)\nlts__t_bytes  Counter  bytes\n")


def test_profile_missing_a_launch_is_refused() -> None:
    with pytest.raises(ProfileError, match="expected 2 profiled launches per kernel"):
        measured_launches(parse_launches(str(profile(launches=1)["csv"])))


def test_a10_ratio_range_is_the_per_call_extremes() -> None:
    machine = build_a10_machine(
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


def test_a10_entry_with_another_chip_name_fails_the_contract() -> None:
    calls = [call(0, "NVIDIA A10G"), call(1, "NVIDIA A10G"), call(2, "NVIDIA A10G")]
    machine = build_a10_machine(4096, calls, profile(), SPEED_KERNELS, True, COMMIT)
    doc = {"version": 1, "source": {"commit": COMMIT, "code": ""}, "machines": [machine]}
    assert problems(doc, SPEED_KERNELS) == ["machines[0]: name must be 'NVIDIA A10'"]


def test_calls_on_different_hardware_are_a_run_problem() -> None:
    other = call(2)
    other["gpu"] = {**other["gpu"], "power_limit_w": 150.0}
    assert (
        run_problem([call(0), call(1), other], profile(), COMMIT)
        == "calls ran with different gpu facts"
    )


def write_run(monkeypatch: pytest.MonkeyPatch, folder: Path, vendor: str, tensor: int = 0) -> None:
    monkeypatch.setattr(driver, "ROOT", folder)
    run_profile = profile(vendor, tensor=tensor)
    launches = parse_launches(str(run_profile["csv"]))
    calls = [call(0), call(1), call(2)]
    write_a10(folder, 4096, calls, run_profile, [], launches, COMMIT, True)


def test_a10_run_keeps_the_mac_entry_and_writes_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mac = {"id": "m3-pro", "versions": {"commit": "b" * 40}}
    (tmp_path / "speeds.json").write_text(json.dumps({"machines": [mac]}))
    write_run(monkeypatch, tmp_path, CUBLAS)
    speeds = json.loads((tmp_path / "speeds.json").read_text())
    assert [(m["id"], m["versions"]["commit"]) for m in speeds["machines"]] == [
        ("m3-pro", "b" * 40),
        ("a10", COMMIT),
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


def test_simt_vendor_kernel_with_no_tensor_instructions_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_run(monkeypatch, tmp_path, CUBLAS)
    sidecar = json.loads((tmp_path / "a10-runs.json").read_text())
    assert sidecar["vendorKernel"] == {"name": CUBLAS, "tensorMetric": TENSOR, "tensorValue": 0.0}
    assert sidecar_problems(sidecar) == []


def test_tensor_op_vendor_kernel_fails_the_bytes_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_run(monkeypatch, tmp_path, TENSOR_OP)
    doc = json.loads((tmp_path / "bytes.json").read_text())
    assert bytes_problems(doc) == [f"vendor kernel {TENSOR_OP!r} has 'tensorop' in its name"]


@pytest.mark.parametrize(
    ("name", "part"),
    [
        (TENSOR_OP, "tensorop"),
        ("sm80_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize128x128x16", "tf32"),
        ("ampere_s16816gemm_fp32_128x128_ldg8_nn", "s16816"),
        ("volta_sgemm_HMMA_128x64_nn", "hmma"),
    ],
)
def test_vendor_kernel_with_a_tensor_core_name_stops_the_run_before_out(
    name: str, part: str
) -> None:
    launches = parse_launches(str(profile(name)["csv"]))
    assert profile_problem(launches, TENSOR) == f"vendor kernel {name!r} has {part!r} in its name"


def test_vendor_kernel_that_ran_tensor_instructions_stops_the_run_before_out() -> None:
    launches = parse_launches(str(profile(tensor=512)["csv"]))
    assert (
        profile_problem(launches, TENSOR)
        == f"vendor kernel {CUBLAS!r} used the tensor pipe: {TENSOR} = 512.0"
    )


def test_sidecar_with_tensor_instructions_fails_the_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_run(monkeypatch, tmp_path, CUBLAS, tensor=512)
    sidecar = json.loads((tmp_path / "a10-runs.json").read_text())
    assert sidecar_problems(sidecar) == [
        f"vendor kernel {CUBLAS!r} used the tensor pipe: {TENSOR} = 512.0"
    ]
