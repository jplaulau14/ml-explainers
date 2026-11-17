from typing import TypedDict

VERSION = 1
SPEED_KERNELS = frozenset({"naive", "tile32", "reg4x4", "vendor"})
CHALLENGE_KERNELS = frozenset({"reg8x8"})
MACHINE_IDS = frozenset({"m3-pro", "a10"})
A10_NAME = "NVIDIA A10"
MIN_RUNS = 3
SOURCE_KEYS = {"commit", "code"}
MACHINE_KEYS = {"id", "name", "n", "runs", "conditions", "versions", "kernels"}
KERNEL_KEYS = {"kernel", "ratioMin", "ratioMax", "gflopsThisRun"}
BYTES_KEYS = {"version", "source", "gpu", "n", "vendorKernel", "rows"}
BYTES_ROW_KEYS = {"kernel", "globalLoadRequests", "sharedLoads", "l2Bytes", "dramReadBytes"}
VENDOR_KEYS = {"name", "tensorMetric", "tensorValue"}
TENSOR_NAME_PARTS = ("tensorop", "tf32", "hmma", "xmma", "s1688", "s16816", "s884")

Condition = str | int | float | bool


class Source(TypedDict):
    commit: str
    code: str


class KernelRow(TypedDict):
    kernel: str
    ratioMin: float
    ratioMax: float
    gflopsThisRun: float


class Machine(TypedDict):
    id: str
    name: str
    n: int
    runs: int
    conditions: dict[str, Condition]
    versions: dict[str, str]
    kernels: list[KernelRow]


class SpeedsDoc(TypedDict):
    version: int
    source: Source
    machines: list[Machine]


class BytesRow(TypedDict):
    kernel: str
    globalLoadRequests: int
    sharedLoads: int
    l2Bytes: int
    dramReadBytes: int


class BytesDoc(TypedDict):
    version: int
    source: Source
    gpu: str
    n: int
    vendorKernel: str
    rows: list[BytesRow]


def is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def kernel_problems(row: object, where: str) -> list[str]:
    if not isinstance(row, dict) or set(row) != KERNEL_KEYS:
        return [f"{where}: kernel row keys must be {sorted(KERNEL_KEYS)}"]
    if not all(is_number(row[key]) for key in ("ratioMin", "ratioMax", "gflopsThisRun")):
        return [f"{where}: ratios and GFLOP/s must be positive numbers"]
    if row["ratioMin"] > row["ratioMax"]:
        return [f"{where}: ratioMin is above ratioMax"]
    if row["kernel"] == "naive" and not row["ratioMin"] == row["ratioMax"] == 1:
        return [f"{where}: naive must have ratio 1"]
    return []


def is_count(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value >= 0


def vendor_name_problem(name: object) -> str | None:
    if not isinstance(name, str) or not name:
        return "vendor kernel name must be a non-empty string"
    denied = [part for part in TENSOR_NAME_PARTS if part in name.lower()]
    return f"vendor kernel {name!r} has {denied[0]!r} in its name" if denied else None


def vendor_kernel_problem(name: object, tensor: object, value: object) -> str | None:
    problem = vendor_name_problem(name)
    if problem:
        return problem
    if not isinstance(tensor, str) or not tensor:
        return f"vendor kernel {name!r} has no tensor-pipe metric"
    if not is_count(value):
        return f"vendor kernel {name!r} has no {tensor} value"
    if value != 0:
        return f"vendor kernel {name!r} used the tensor pipe: {tensor} = {value}"
    return None


def sidecar_problems(doc: object) -> list[str]:
    vendor = doc.get("vendorKernel") if isinstance(doc, dict) else None
    if not isinstance(vendor, dict) or set(vendor) != VENDOR_KEYS:
        return [f"vendorKernel keys must be {sorted(VENDOR_KEYS)}"]
    problem = vendor_kernel_problem(vendor["name"], vendor["tensorMetric"], vendor["tensorValue"])
    return [problem] if problem else []


def shape_problems(machine: dict[str, object], where: str) -> list[str]:
    conditions, versions = machine["conditions"], machine["versions"]
    if not isinstance(conditions, dict) or not isinstance(versions, dict):
        return [f"{where}: conditions and versions must be objects"]
    found = [
        f"{where}: conditions.{key} must be a string, number or boolean"
        for key, value in conditions.items()
        if not isinstance(value, Condition)
    ]
    found += [
        f"{where}: versions.{key} must be a non-empty string"
        for key, value in versions.items()
        if not isinstance(value, str) or not value
    ]
    if "commit" not in versions:
        found.append(f"{where}: versions.commit is missing")
    return found


def a10_problems(machine: dict[str, object], where: str) -> list[str]:
    found = [] if machine["name"] == A10_NAME else [f"{where}: name must be {A10_NAME!r}"]
    conditions = machine["conditions"]
    if isinstance(conditions, dict) and conditions.get("publication") is not True:
        found.append(f"{where}: conditions.publication must be true")
    return found


def mac_problems(machine: dict[str, object], where: str) -> list[str]:
    conditions = machine["conditions"]
    if not isinstance(conditions, dict):
        return []
    found = []
    if conditions.get("acPower") is not True:
        found.append(f"{where}: conditions.acPower must be true")
    if conditions.get("lowPowerMode") is not False:
        found.append(f"{where}: conditions.lowPowerMode must be false")
    if conditions.get("publication") is not True:
        found.append(f"{where}: conditions.publication must be true")
    return found


def machine_problems(machine: object, kernel_ids: frozenset[str], where: str) -> list[str]:
    if not isinstance(machine, dict) or set(machine) != MACHINE_KEYS:
        return [f"{where}: machine keys must be {sorted(MACHINE_KEYS)}"]
    found = [] if machine["id"] in MACHINE_IDS else [f"{where}: unknown machine id"]
    runs = machine["runs"]
    if not isinstance(runs, int) or runs < MIN_RUNS:
        found.append(f"{where}: needs at least {MIN_RUNS} runs")
    rows = machine["kernels"] if isinstance(machine["kernels"], list) else []
    ids = [row.get("kernel") if isinstance(row, dict) else None for row in rows]
    if sorted(map(str, ids)) != sorted(kernel_ids):
        found.append(f"{where}: kernels must be {sorted(kernel_ids)}, got {ids}")
    for kernel, row in zip(ids, rows, strict=True):
        found += kernel_problems(row, f"{where}.{kernel}")
    found += shape_problems(machine, where)
    if machine["id"] == "m3-pro":
        found += mac_problems(machine, where)
    if machine["id"] == "a10":
        found += a10_problems(machine, where)
    return found


def problems(doc: object, kernel_ids: frozenset[str]) -> list[str]:
    if not isinstance(doc, dict) or set(doc) != {"version", "source", "machines"}:
        return ["document keys must be ['machines', 'source', 'version']"]
    found = [] if doc["version"] == VERSION else [f"version must be {VERSION}"]
    source = doc["source"]
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        found.append(f"source keys must be {sorted(SOURCE_KEYS)}")
    machines = doc["machines"] if isinstance(doc["machines"], list) else []
    if not machines:
        found.append("machines must be a non-empty list")
    ids = [m.get("id") for m in machines if isinstance(m, dict)]
    if len(ids) != len(set(ids)):
        found.append("machine ids must be unique")
    for index, machine in enumerate(machines):
        found += machine_problems(machine, kernel_ids, f"machines[{index}]")
    return found


def bytes_row_problems(row: object, where: str) -> list[str]:
    if not isinstance(row, dict) or set(row) != BYTES_ROW_KEYS:
        return [f"{where}: row keys must be {sorted(BYTES_ROW_KEYS)}"]
    counts = BYTES_ROW_KEYS - {"kernel"}
    return [f"{where}.{key} must be a count" for key in sorted(counts) if not is_count(row[key])]


def bytes_problems(doc: object) -> list[str]:
    if not isinstance(doc, dict) or set(doc) != BYTES_KEYS:
        return [f"document keys must be {sorted(BYTES_KEYS)}"]
    found = [] if doc["version"] == VERSION else [f"version must be {VERSION}"]
    source = doc["source"]
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        found.append(f"source keys must be {sorted(SOURCE_KEYS)}")
    if doc["gpu"] != A10_NAME:
        found.append(f"gpu must be {A10_NAME!r}")
    if not isinstance(doc["n"], int) or doc["n"] <= 0:
        found.append("n must be a positive integer")
    found += [problem for problem in [vendor_name_problem(doc["vendorKernel"])] if problem]
    rows = doc["rows"] if isinstance(doc["rows"], list) else []
    ids = [row.get("kernel") if isinstance(row, dict) else None for row in rows]
    if sorted(map(str, ids)) != sorted(SPEED_KERNELS):
        found.append(f"rows must be {sorted(SPEED_KERNELS)}, got {ids}")
    for kernel, row in zip(ids, rows, strict=True):
        found += bytes_row_problems(row, f"rows.{kernel}")
    return found
