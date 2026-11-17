import json
from collections.abc import Mapping
from pathlib import Path

from tiling_explainer.a10.ncu import (
    DURATION,
    L1_HIT_RATE,
    SHARED_WAVEFRONTS,
    Launch,
    bytes_rows,
    measured_launches,
)
from tiling_explainer.a10.spec import (
    KERNEL_ORDER,
    SAMPLES,
    TIMING_METHOD,
    WARMUPS,
)
from tiling_explainer.contract import (
    CHALLENGE_KERNELS,
    SPEED_KERNELS,
    VERSION,
    BytesDoc,
    Condition,
    Machine,
    Source,
)
from tiling_explainer.driver import merged, write_json
from tiling_explainer.provenance import code_url
from tiling_explainer.ratios import ratios_to, summarize

Call = Mapping[str, object]


def section(call: Call, key: str) -> dict[str, Condition]:
    value = call[key]
    if not isinstance(value, dict):
        raise TypeError(f"call field {key} must be an object")
    return value


def timings(call: Call) -> dict[str, float]:
    return {kernel: float(row["ms"]) for kernel, row in section(call, "kernels").items()}


def distinct_gpus(calls: list[Call]) -> int:
    return len({call["gpuUuid"] for call in calls})


def run_problem(calls: list[Call], profile: Call, commit: str) -> str | None:
    every = [*calls, profile]
    if any(call["commit"] != commit for call in every):
        return "a call recorded a different commit"
    for key in ("gpu", "software"):
        facts = {json.dumps({**section(c, key), "uuid": ""}, sort_keys=True) for c in every}
        if len(facts) > 1:
            return f"calls ran with different {key} facts"
    return None


def conditions(calls: list[Call], publication: bool) -> dict[str, Condition]:
    gpu = section(calls[0], "gpu")
    return {
        "publication": publication,
        "dtype": "float32",
        "timingMethod": TIMING_METHOD,
        "timingWarmups": WARMUPS,
        "timingLaunchesPerSample": 1,
        "timingSamples": SAMPLES,
        "maxSmClockMHz": gpu["max_sm_clock_mhz"],
        "powerLimitW": gpu["power_limit_w"],
        "smCount": gpu["smCount"],
        "l2Bytes": gpu["l2Bytes"],
        "distinctGpus": distinct_gpus(calls),
    }


def versions(calls: list[Call], profile: Call, commit: str) -> dict[str, str]:
    gpu, software = section(calls[0], "gpu"), section(calls[0], "software")
    return {
        "commit": commit,
        "gpu": str(gpu["name"]),
        "driver": str(gpu["driver"]),
        "cudaRuntime": str(software["cudaRuntime"]),
        "cudaDriver": str(software["cudaDriver"]),
        "cupy": str(software["cupy"]),
        "ncu": str(profile["ncu"]),
        "modal": str(software["modal"]),
        "python": str(software["python"]),
    }


def build_a10_machine(
    n: int,
    calls: list[Call],
    profile: Call,
    kernels: frozenset[str],
    publication: bool,
    commit: str,
) -> Machine:
    runs = [timings(call) for call in calls]
    return {
        "id": "a10",
        "name": str(section(calls[0], "gpu")["name"]),
        "n": n,
        "runs": len(runs),
        "conditions": conditions(calls, publication),
        "versions": versions(calls, profile, commit),
        "kernels": [summarize(kernel, n, runs) for kernel in KERNEL_ORDER if kernel in kernels],
    }


def build_bytes(n: int, source: Source, gpu: str, measured: dict[str, Launch]) -> BytesDoc:
    return {
        "version": VERSION,
        "source": source,
        "gpu": gpu,
        "n": n,
        "vendorKernel": str(measured["vendor"]["name"]),
        "rows": bytes_rows(measured),
    }


def profile_detail(
    profile: Call, launches: list[Launch], measured: dict[str, Launch]
) -> dict[str, object]:
    extra = (SHARED_WAVEFRONTS, L1_HIT_RATE, DURATION, str(profile["tensorMetric"]))
    per_kernel = {kernel: {m: launch[m] for m in extra} for kernel, launch in measured.items()}
    keep = ("container", "gpuUuid", "ncu", "command", "tensorMetric", "gpu", "software")
    return {**{key: profile[key] for key in keep}, "measured": per_kernel, "launches": launches}


def vendor_kernel(profile: Call, launch: Launch) -> dict[str, object]:
    tensor = str(profile["tensorMetric"])
    return {"name": launch["name"], "tensorMetric": tensor, "tensorValue": launch[tensor]}


def call_detail(call: Call) -> dict[str, object]:
    keep = ("container", "gpuUuid", "gpu", "software", "kernels", "attributes")
    return {
        **{key: call[key] for key in keep},
        "ratiosToNaive": ratios_to("naive", timings(call)),
        "ratiosToVendor": ratios_to("vendor", timings(call)),
    }


def write_a10(
    folder: Path,
    n: int,
    calls: list[Call],
    profile: Call,
    refusals: list[Call],
    launches: list[Launch],
    commit: str,
    publication: bool,
) -> None:
    source: Source = {"commit": commit, "code": code_url(commit)}
    measured = measured_launches(launches)
    for name, kernels in (("speeds.json", SPEED_KERNELS), ("challenge.json", CHALLENGE_KERNELS)):
        machine = build_a10_machine(n, calls, profile, kernels, publication, commit)
        write_json(folder / name, merged(folder / name, source, machine))
    gpu = str(section(calls[0], "gpu")["name"])
    write_json(folder / "bytes.json", build_bytes(n, source, gpu, measured))
    sidecar = {
        "version": VERSION,
        "source": source,
        "n": n,
        "publication": publication,
        "distinctGpus": distinct_gpus([*calls, profile]),
        "vendorKernel": vendor_kernel(profile, measured["vendor"]),
        "calls": [call_detail(call) for call in calls],
        "profile": profile_detail(profile, launches, measured),
        "refusals": refusals,
    }
    write_json(folder / "a10-runs.json", sidecar)
