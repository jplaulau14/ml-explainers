import os
import platform
from importlib.metadata import version
from pathlib import Path

import torch

from mb_explainer.timing import fastest, median, samples

FLAG_NAMES = ("avx2", "avx512f", "avx512_bf16", "amx_bf16")
DTYPES = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16}


def cpu_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields.setdefault(key.strip(), value.strip())
    return fields


def processor_count() -> int:
    text = Path("/proc/cpuinfo").read_text()
    return sum(line.startswith("processor") for line in text.splitlines())


def parse_cache_size(text: str) -> int:
    raw = text.strip()
    if raw.endswith("K"):
        return int(raw[:-1]) * 1024
    if raw.endswith("M"):
        return int(raw[:-1]) * 1024**2
    return int(raw)


def caches() -> list[dict[str, int | str]]:
    root = Path("/sys/devices/system/cpu/cpu0/cache")
    found = []
    for index in sorted(root.glob("index*")):
        found.append(
            {
                "level": int((index / "level").read_text()),
                "bytes": parse_cache_size((index / "size").read_text()),
                "shared": (index / "shared_cpu_list").read_text().strip(),
                "type": (index / "type").read_text().strip(),
            }
        )
    return found


def mem_bytes() -> dict[str, int]:
    found = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        found[key] = int(rest.split()[0]) * 1024
    return found


def l3_bytes(levels: list[dict[str, int | str]]) -> int:
    sizes = [int(row["bytes"]) for row in levels if row["level"] == 3]
    return max(sizes) if sizes else 0


def relevant_flags(text: str) -> list[str]:
    present = set(text.split())
    return [name for name in FLAG_NAMES if name in present]


def configure(cores: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    torch.set_num_threads(cores)


def stream_elements(l3: int, available: int, itemsize: int) -> int:
    target = 4 * l3
    cap = max(int(available * 0.35), itemsize * 3)
    total = min(max(target, 256 * 2**20), cap)
    return max((total // 3) // itemsize, 1)


def _kernel(kind: str, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
    if kind == "copy":
        a.copy_(b)
    else:
        torch.add(b, c, alpha=3.0, out=a)


def run_stream(kind: str, dtype: torch.dtype, elements: int, repeats: int) -> dict:
    a = torch.empty(elements, dtype=dtype)
    b = torch.ones(elements, dtype=dtype)
    c = torch.full((elements,), 2.0, dtype=dtype)
    item = a.element_size()
    moved = (2 if kind == "copy" else 3) * elements * item
    took = samples(lambda: _kernel(kind, a, b, c), repeats, 1)
    a[0].item()
    best = fastest(took)
    return {
        "kernel": kind,
        "dtype": str(dtype).removeprefix("torch."),
        "elements": elements,
        "arrayBytes": elements * item,
        "workingSetBytes": (2 if kind == "copy" else 3) * elements * item,
        "bytesPerCall": moved,
        "repeats": repeats,
        "seconds": took,
        "medianSeconds": median(took),
        "bestSeconds": best,
        "medianBytesPerSecond": moved / median(took),
        "bestBytesPerSecond": moved / best,
    }


def run_peak(dtype_name: str, n: int, repeats: int) -> dict:
    dtype = DTYPES[dtype_name]
    try:
        left = torch.randn(n, n, dtype=dtype)
        right = torch.randn(n, n, dtype=dtype)
        out = torch.empty(n, n, dtype=dtype)
        took = samples(lambda: torch.mm(left, right, out=out), repeats, 1)
        out[0, 0].item()
    except Exception as exc:
        return {
            "dtype": dtype_name,
            "n": n,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    flops = 2 * n**3
    best = fastest(took)
    return {
        "dtype": dtype_name,
        "n": n,
        "status": "ok",
        "flops": flops,
        "repeats": repeats,
        "seconds": took,
        "medianSeconds": median(took),
        "bestSeconds": best,
        "medianFlopsPerSecond": flops / median(took),
        "bestFlopsPerSecond": flops / best,
    }


def library_versions() -> dict[str, str]:
    names = ("numpy", "torch", "transformers")
    return {name: version(name) for name in names} | {"python": platform.python_version()}


def cpu_block(fields: dict[str, str], cores: int, levels: list[dict]) -> dict:
    info = os.uname()
    return {
        "model": fields.get("model name", ""),
        "vendor": fields.get("vendor_id", ""),
        "family": int(fields.get("cpu family", "0")),
        "modelId": int(fields.get("model", "0")),
        "stepping": int(fields.get("stepping", "0")),
        "cores": cores,
        "flags": relevant_flags(fields.get("flags", "")),
        "caches": levels,
        "l3Bytes": l3_bytes(levels),
        "uname": {"system": info.sysname, "release": info.release, "machine": info.machine},
    }


def probe(repeats: int, peak_n: int, peak_repeats: int, peak_dtypes: tuple[str, ...]) -> dict:
    fields = cpu_fields()
    cores = processor_count()
    levels = caches()
    memory = mem_bytes()
    configure(cores)
    l3 = l3_bytes(levels)
    available = memory["MemAvailable"]
    streams = []
    for kind, dtype_name in (("copy", "float64"), ("triad", "float64"), ("triad", "float32")):
        dtype = DTYPES[dtype_name]
        elements = stream_elements(l3, available, torch.empty(1, dtype=dtype).element_size())
        row = run_stream(kind, dtype, elements, repeats)
        row["exceedsL3"] = row["workingSetBytes"] > l3
        streams.append(row)
    peaks = [run_peak(name, peak_n, peak_repeats) for name in peak_dtypes]
    return {
        "cpu": cpu_block(fields, cores, levels),
        "memory": {"totalBytes": memory["MemTotal"], "availableBytes": available},
        "libraries": library_versions(),
        "threads": cores,
        "streams": streams,
        "peaks": peaks,
    }


def triad(machine: dict) -> dict:
    return next(
        row for row in machine["streams"] if row["kernel"] == "triad" and row["dtype"] == "float64"
    )


def float32_peak(machine: dict) -> dict:
    return next(
        row for row in machine["peaks"] if row["dtype"] == "float32" and row["status"] == "ok"
    )
