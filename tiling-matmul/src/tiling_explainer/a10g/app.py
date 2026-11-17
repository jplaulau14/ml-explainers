import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import modal

import tiling_explainer
from tiling_explainer.a10g.ncu import ncu_command
from tiling_explainer.a10g.spec import SMI_QUERY, SmiReading, guarded

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
app = modal.App("tiling-matmul-a10g", image=image)

with image.imports():
    from tiling_explainer.a10g import bench

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
        "container": os.environ.get("MODAL_TASK_ID", "unknown"),
        "gpuUuid": reading.uuid,
        "gpu": {**asdict(reading), **bench.device_facts()},
        "software": software,
    }


def time_on_a10g(reading: SmiReading, n: int, commit: str) -> Call:
    facts = call_facts(reading, commit)
    return {**facts, "kernels": bench.time_kernels(n), "attributes": bench.kernel_attributes()}


def profile_on_a10g(reading: SmiReading, n: int, commit: str) -> Call:
    facts = call_facts(reading, commit)
    command = ncu_command(sys.executable, n)
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
    return {**facts, "ncu": version, "command": " ".join(command), "csv": done.stdout}


@app.function(gpu="A10G", timeout=CALL_TIMEOUT_SECONDS, single_use_containers=True)
def timing_call(n: int, commit: str) -> Call:
    return guarded(run_text(SMI_QUERY), lambda reading: time_on_a10g(reading, n, commit))


@app.function(gpu="A10G", timeout=CALL_TIMEOUT_SECONDS, single_use_containers=True)
def profile_call(n: int, commit: str) -> Call:
    return guarded(run_text(SMI_QUERY), lambda reading: profile_on_a10g(reading, n, commit))


def run_calls(n: int, calls: int, commit: str) -> tuple[list[Call], Call]:
    with modal.enable_output(), app.run():
        timings = [timing_call.spawn(n, commit) for _ in range(calls)]
        profile = profile_call.spawn(n, commit)
        return [timing.get() for timing in timings], profile.get()
