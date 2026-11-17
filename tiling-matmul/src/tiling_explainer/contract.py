from typing import TypedDict

VERSION = 1
SPEED_KERNELS = frozenset({"naive", "tile32", "reg4x4", "vendor"})
CHALLENGE_KERNELS = frozenset({"reg8x8"})
MACHINE_IDS = frozenset({"m3-pro", "a10g"})
MIN_RUNS = 3
SOURCE_KEYS = {"commit", "code"}
MACHINE_KEYS = {"id", "name", "n", "runs", "conditions", "versions", "kernels"}
KERNEL_KEYS = {"kernel", "ratioMin", "ratioMax", "gflopsThisRun"}


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
    conditions: dict[str, object]
    versions: dict[str, str]
    kernels: list[KernelRow]


class SpeedsDoc(TypedDict):
    version: int
    source: Source
    machines: list[Machine]


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


def mac_problems(machine: dict[str, object], where: str) -> list[str]:
    conditions = machine["conditions"]
    if not isinstance(conditions, dict):
        return [f"{where}: conditions must be an object"]
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
    if machine["id"] == "m3-pro":
        found += mac_problems(machine, where)
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
