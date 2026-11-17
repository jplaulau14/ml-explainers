import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import modal

import tiling_explainer
from tiling_explainer.a10.gather import gather
from tiling_explainer.a10.ncu import ncu_command, tensor_metrics
from tiling_explainer.a10.spec import SMI_QUERY, SmiReading, guarded
from tiling_explainer.a10.upload import code_record, uploaded_files

PROJECT = Path(__file__).resolve().parents[3]
CUDA_IMAGE = (
    "nvidia/cuda:12.9.0-devel-ubuntu22.04"
    "@sha256:26cd7d00c09c1f3ff187929c7f4aa7e9273020cb18e86730f40e062e8ffa4d33"
)
UV_VERSION = "0.13.1"
CALL_TIMEOUT_SECONDS = 900
COMMAND_TIMEOUT_SECONDS = 600

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .uv_sync(str(PROJECT), groups=["gpu"], extra_options="--no-dev", uv_version=UV_VERSION)
    .env({"CUPY_TF32": "0"})
    .add_local_python_source("tiling_explainer")
)
app = modal.App("tiling-matmul-a10", image=image)

with image.imports():
    from tiling_explainer.a10 import bench

Call = dict[str, object]


def run_text(args: list[str]) -> str:
    done = subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS
    )
    return done.stdout


def call_facts(reading: SmiReading, commit: str) -> Call:
    software = {**bench.software_versions(), "modal": modal.__version__}
    return {
        "commit": commit,
        "container": container(),
        "gpuUuid": reading.uuid,
        "code": code_record(uploaded_files(Path(tiling_explainer.__file__).parent)),
        "gpu": {**asdict(reading), **bench.device_facts()},
        "software": software,
    }


def time_on_a10(reading: SmiReading, n: int, commit: str) -> Call:
    facts = call_facts(reading, commit)
    return {**facts, "kernels": bench.time_kernels(n), "attributes": bench.kernel_attributes()}


def profile_on_a10(reading: SmiReading, n: int, commit: str) -> Call:
    facts = call_facts(reading, commit)
    tensors = tensor_metrics(run_text(["ncu", "--query-metrics"]))
    command = ncu_command(sys.executable, n, tensors)
    package_parent = str(Path(tiling_explainer.__file__).resolve().parents[1])
    done = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        env={**os.environ, "PYTHONPATH": package_parent},
    )
    if done.returncode != 0:
        raise RuntimeError(f"ncu failed with status {done.returncode}:\n{done.stderr[-3000:]}")
    version = run_text(["ncu", "--version"]).strip().splitlines()[-1]
    detail = {"ncu": version, "command": " ".join(command), "tensorMetrics": tensors}
    return {**facts, **detail, "csv": done.stdout}


def container() -> str:
    return os.environ.get("MODAL_TASK_ID", "unknown")


@app.function(gpu="A10", timeout=CALL_TIMEOUT_SECONDS, single_use_containers=True)
def timing_call(n: int, commit: str) -> Call:
    result = guarded(run_text(SMI_QUERY), lambda reading: time_on_a10(reading, n, commit))
    return {"container": container(), **result}


@app.function(gpu="A10", timeout=CALL_TIMEOUT_SECONDS, single_use_containers=True)
def profile_call(n: int, commit: str) -> Call:
    result = guarded(run_text(SMI_QUERY), lambda reading: profile_on_a10(reading, n, commit))
    return {"container": container(), **result}


def run_calls(n: int, calls: int, commit: str) -> tuple[list[Call], Call, list[Call]]:
    starters = {
        "timing": lambda: timing_call.spawn(n, commit),
        "profile": lambda: profile_call.spawn(n, commit),
    }
    with modal.enable_output(), app.run():
        accepted, refusals = gather(starters, {"timing": calls, "profile": 1})
    return accepted["timing"], accepted["profile"][0], refusals
