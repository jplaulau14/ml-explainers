from collections.abc import Callable
from dataclasses import asdict, dataclass

from tiling_explainer.contract import A10_NAME

WARMUPS = 3
SAMPLES = 10
RTOL = 1e-4
ATOL = 1e-3
SEED = 0
KERNEL_ORDER = ("naive", "tile32", "reg4x4", "reg8x8", "vendor")
PROFILED = ("naive", "tile32", "reg4x4", "vendor")
TIMING_METHOD = "CUDA events around each launch"
SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,driver_version,clocks.max.sm,power.limit,uuid",
    "--format=csv,noheader,nounits",
]


class WrongChip(RuntimeError):
    pass


@dataclass(frozen=True)
class SmiReading:
    name: str
    driver: str
    max_sm_clock_mhz: int
    power_limit_w: float
    uuid: str


def parse_smi(text: str) -> SmiReading:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise WrongChip(f"expected one GPU, nvidia-smi listed {len(lines)}")
    name, driver, clock, power, uuid = (field.strip() for field in lines[0].split(","))
    return SmiReading(name, driver, int(float(clock)), float(power), uuid)


def guarded(smi_text: str, work: Callable[[SmiReading], dict[str, object]]) -> dict[str, object]:
    reading = parse_smi(smi_text)
    if reading.name != A10_NAME:
        refused = f"GPU is {reading.name!r}, not {A10_NAME!r}; nothing was timed"
        return {"refused": refused, "gpu": asdict(reading)}
    return work(reading)
