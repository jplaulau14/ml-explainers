import platform
import re
import subprocess
from dataclasses import dataclass
from importlib.metadata import version

LOW_POWER = re.compile(r"^\s*lowpowermode\s+(\d)\s*$", re.MULTILINE)
DRAWING_FROM = re.compile(r"Now drawing from '([^']+)'")


@dataclass(frozen=True)
class PowerState:
    ac_power: bool
    low_power_mode: bool

    def to_json(self) -> dict[str, bool]:
        return {"acPower": self.ac_power, "lowPowerMode": self.low_power_mode}


def parse_power(settings: str, battery: str) -> PowerState:
    low_power = LOW_POWER.search(settings)
    source = DRAWING_FROM.search(battery)
    if low_power is None or source is None:
        raise ValueError("pmset output has no lowpowermode line or power source")
    return PowerState(
        ac_power=source.group(1) == "AC Power", low_power_mode=low_power.group(1) != "0"
    )


def run_text(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout


def read_power() -> PowerState:
    return parse_power(run_text(["pmset", "-g"]), run_text(["pmset", "-g", "batt"]))


def chip_name() -> str:
    return run_text(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()


def machine_id(chip: str) -> str:
    return chip.removeprefix("Apple ").lower().replace(" ", "-")


def software_versions() -> dict[str, str]:
    return {
        "macos": platform.mac_ver()[0],
        "mlx": version("mlx"),
        "python": platform.python_version(),
    }


def refusal(states: list[PowerState], dirty: bool) -> str | None:
    if dirty:
        return "uncommitted changes in tiling-matmul/ outside out/"
    if not all(state.ac_power for state in states):
        return "running on battery"
    if any(state.low_power_mode for state in states):
        return "Low Power Mode is on"
    return None
