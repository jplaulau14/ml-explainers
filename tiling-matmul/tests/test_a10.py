import json
from collections.abc import Callable
from pathlib import Path

import pytest
from a10_fakes import (
    CODE,
    COMMIT,
    CUBLAS,
    HMMA,
    HMMA_CYCLES,
    TENSOR_OP,
    TENSORS,
    call,
    profile,
)

from tiling_explainer.a10.gather import TooManyRefusals, gather
from tiling_explainer.a10.ncu import (
    L2_BYTES,
    ProfileError,
    measured_launches,
    parse_launches,
    tensor_metrics,
)
from tiling_explainer.a10.outputs import a10_documents, build_a10_machine, run_problem
from tiling_explainer.a10.run import profile_problem
from tiling_explainer.a10.spec import guarded
from tiling_explainer.contract import SPEED_KERNELS, bytes_problems, problems, sidecar_problems

QUERY = (Path(__file__).parent / "fixtures" / "a10_query_metrics.txt").read_text()


def smi(name: str) -> str:
    return f"{name}, 620.70.06, 1710, 300.00, GPU-1234\n"


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


def test_every_tensor_pipe_counter_the_device_lists_is_collected() -> None:
    assert tensor_metrics(QUERY) == sorted(
        f"{unit}__{metric}.sum"
        for unit in ("sm", "smsp")
        for metric in (
            "inst_executed_pipe_tensor",
            "inst_executed_pipe_tensor_op_hmma",
            "inst_executed_pipe_tensor_op_imma",
            "pipe_tensor_cycles_active",
            "pipe_tensor_op_hmma_cycles_active",
            "pipe_tensor_op_imma_cycles_active",
        )
    )


def test_device_without_a_tensor_pipe_counter_fails_the_profile() -> None:
    with pytest.raises(ProfileError, match="lists no pipe_tensor counter"):
        tensor_metrics("Device NVIDIA A10 (GA102)\nlts__t_bytes  Counter  byte  bytes\n")


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


def test_a10_entry_below_publication_size_fails_the_contract() -> None:
    machine = build_a10_machine(
        1024, [call(0), call(1), call(2)], profile(), SPEED_KERNELS, True, COMMIT
    )
    doc = {"version": 1, "source": {"commit": COMMIT, "code": ""}, "machines": [machine]}
    assert problems(doc, SPEED_KERNELS) == ["machines[0]: n must be 4096"]


def test_call_on_another_chip_is_a_run_problem() -> None:
    calls = [call(0), call(1, "NVIDIA A10G"), call(2)]
    assert run_problem(calls, profile(), CODE) == "a call ran on ['NVIDIA A10G'], not 'NVIDIA A10'"


def test_call_that_ran_an_extra_file_is_a_run_problem() -> None:
    other = call(2)
    other["code"] = {"files": [*CODE["files"], "stray.py"], "digest": "0" * 64}
    assert run_problem([call(0), call(1), other], profile(), CODE) == (
        "a call ran a different file set than HEAD: extra ['stray.py'], missing []"
    )


def test_calls_on_different_hardware_are_a_run_problem() -> None:
    other = call(2)
    other["gpu"] = {**other["gpu"], "power_limit_w": 150.0}
    assert (
        run_problem([call(0), call(1), other], profile(), CODE)
        == "calls ran with different gpu facts"
    )


def documents(folder: Path, vendor: str = CUBLAS, cycles: int = 0) -> dict[str, dict]:
    run_profile = profile(vendor, cycles=cycles)
    launches = parse_launches(str(run_profile["csv"]))
    calls = [call(0), call(1), call(2)]
    built = a10_documents(folder, 4096, calls, run_profile, [], launches, COMMIT, True)
    return {path.name: document for path, document in built.items()}


def test_a10_run_keeps_the_mac_entry_and_builds_bytes(tmp_path: Path) -> None:
    mac = {"id": "m3-pro", "versions": {"commit": "b" * 40}}
    (tmp_path / "speeds.json").write_text(json.dumps({"machines": [mac]}))
    built = documents(tmp_path)
    assert [(m["id"], m["versions"]["commit"]) for m in built["speeds.json"]["machines"]] == [
        ("m3-pro", "b" * 40),
        ("a10", COMMIT),
    ]
    assert bytes_problems(built["bytes.json"]) == []
    assert built["bytes.json"]["rows"][0] == {
        "kernel": "naive",
        "globalLoadRequests": 1001,
        "sharedLoads": 1001,
        "l2Bytes": 1001,
        "dramReadBytes": 1001,
    }


def test_sidecar_records_every_tensor_metric_and_counts_timing_gpus_only(tmp_path: Path) -> None:
    sidecar = documents(tmp_path)["a10-runs.json"]
    assert sidecar["vendorKernel"] == {
        "name": CUBLAS,
        "tensorMetrics": {HMMA: 0.0, HMMA_CYCLES: 0.0},
    }
    assert sidecar["distinctGpus"] == 3
    assert sidecar_problems(sidecar, documents(tmp_path)["bytes.json"]) == []


def test_sidecar_keeps_only_the_query_header_and_tensor_pipe_rows(tmp_path: Path) -> None:
    kept = documents(tmp_path)["a10-runs.json"]["profile"]["metricsQuery"].splitlines()
    assert [line.split()[0] for line in kept] == [
        "Device",
        "Metric",
        "sm__inst_executed_pipe_tensor_op_hmma",
        "smsp__average_inst_executed_pipe_tensor_op_hmma_per_warp",
    ]


def test_bytes_from_another_commit_fails_the_sidecar_contract(tmp_path: Path) -> None:
    built = documents(tmp_path)
    other = {**built["bytes.json"], "source": {"commit": "b" * 40, "code": ""}}
    assert sidecar_problems(built["a10-runs.json"], other) == [
        "bytes.json and a10-runs.json must come from the same commit"
    ]


def test_tensor_op_vendor_kernel_fails_the_bytes_contract(tmp_path: Path) -> None:
    assert bytes_problems(documents(tmp_path, TENSOR_OP)["bytes.json"]) == [
        f"vendor kernel {TENSOR_OP!r} has 'tensorop' in its name"
    ]


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
    assert profile_problem(launches, TENSORS) == f"vendor kernel {name!r} has {part!r} in its name"


def test_simt_xmma_name_with_no_tensor_activity_passes() -> None:
    launches = parse_launches(str(profile("sm80_xmma_gemm_f32f32_f32f32_f32_nn_n")["csv"]))
    assert profile_problem(launches, TENSORS) is None


def test_vendor_kernel_with_any_tensor_activity_stops_the_run_before_out() -> None:
    launches = parse_launches(str(profile(cycles=512)["csv"]))
    assert profile_problem(launches, TENSORS) == (
        f"vendor kernel {CUBLAS!r} used the tensor pipe: {{{HMMA_CYCLES!r}: 512.0}}"
    )


def test_sidecar_with_tensor_activity_fails_the_contract(tmp_path: Path) -> None:
    built = documents(tmp_path, cycles=512)
    assert sidecar_problems(built["a10-runs.json"], built["bytes.json"]) == [
        f"vendor kernel {CUBLAS!r} used the tensor pipe: {{{HMMA_CYCLES!r}: 512.0}}"
    ]
