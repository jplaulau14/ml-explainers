import csv
from collections import defaultdict

from tiling_explainer.a10.spec import PROFILED
from tiling_explainer.contract import BytesRow

REQUESTS = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
L2_BYTES = "lts__t_bytes.sum"
DRAM_READ = "dram__bytes_read.sum"
SHARED_LOADS = "smsp__inst_executed_op_shared_ld.sum"
SHARED_WAVEFRONTS = "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
L1_HIT_RATE = "l1tex__t_sector_hit_rate.pct"
DURATION = "gpu__time_duration.sum"
METRICS = (REQUESTS, L2_BYTES, DRAM_READ, SHARED_LOADS, SHARED_WAVEFRONTS, L1_HIT_RATE, DURATION)
TENSOR_PIPE = "pipe_tensor"
HAND_KERNELS = ("naive", "tile32", "reg4x4")
KERNEL_FILTER = "regex:naive|tile32|reg4x4|gemm|Kernel2"
LAUNCHES = 2

Launch = dict[str, str | float]


class ProfileError(RuntimeError):
    pass


def tensor_metrics(query: str) -> list[str]:
    rows = [line.split() for line in query.splitlines()]
    counters = {
        f"{words[0]}.sum"
        for words in rows
        if len(words) > 1 and TENSOR_PIPE in words[0] and words[1] == "Counter"
    }
    if not counters:
        raise ProfileError(f"ncu --query-metrics lists no {TENSOR_PIPE} counter on this device")
    return sorted(counters)


def ncu_command(python: str, n: int, tensors: list[str]) -> list[str]:
    return [
        "ncu",
        "--clock-control",
        "none",
        "--kernel-name-base",
        "demangled",
        "-k",
        KERNEL_FILTER,
        "--metrics",
        ",".join((*METRICS, *tensors)),
        "--print-units",
        "base",
        "--csv",
        python,
        "-m",
        "tiling_explainer.a10.target",
        str(n),
    ]


def parse_launches(text: str) -> list[Launch]:
    lines = [line for line in text.splitlines() if line.startswith('"')]
    launches: dict[str, Launch] = defaultdict(dict)
    for row in csv.DictReader(lines):
        launch = launches[row["ID"]]
        launch["name"] = row["Kernel Name"]
        launch[row["Metric Name"]] = float(row["Metric Value"].replace(",", ""))
    return [launches[key] for key in sorted(launches, key=int)]


def kernel_id(name: str) -> str:
    return name if name in HAND_KERNELS else "vendor"


def measured_launches(launches: list[Launch]) -> dict[str, Launch]:
    grouped: dict[str, list[Launch]] = defaultdict(list)
    for launch in launches:
        grouped[kernel_id(str(launch["name"]))].append(launch)
    counts = {kernel: len(grouped[kernel]) for kernel in PROFILED}
    if any(count != LAUNCHES for count in counts.values()):
        raise ProfileError(f"expected {LAUNCHES} profiled launches per kernel, got {counts}")
    vendor_names = {str(launch["name"]) for launch in grouped["vendor"]}
    if len(vendor_names) != 1:
        raise ProfileError(f"cuBLAS launched different kernels: {sorted(vendor_names)}")
    return {kernel: grouped[kernel][-1] for kernel in PROFILED}


def count(launch: Launch, metric: str) -> int:
    return round(float(launch[metric]))


def bytes_rows(measured: dict[str, Launch]) -> list[BytesRow]:
    return [
        {
            "kernel": kernel,
            "globalLoadRequests": count(launch, REQUESTS),
            "sharedLoads": count(launch, SHARED_LOADS),
            "l2Bytes": count(launch, L2_BYTES),
            "dramReadBytes": count(launch, DRAM_READ),
        }
        for kernel, launch in measured.items()
    ]
