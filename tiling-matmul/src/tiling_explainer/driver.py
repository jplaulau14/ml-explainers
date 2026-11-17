import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tiling_explainer.conditions import (
    PowerState,
    chip_name,
    machine_id,
    read_power,
    refusal,
    software_versions,
)
from tiling_explainer.contract import (
    CHALLENGE_KERNELS,
    SPEED_KERNELS,
    VERSION,
    Machine,
    Source,
    SpeedsDoc,
)
from tiling_explainer.harness import BATCH, ORDER, SAMPLES, WARMUPS
from tiling_explainer.provenance import code_url, head_commit, is_dirty
from tiling_explainer.ratios import gflops, ratios_to, summarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"
RUNS = ROOT / "runs"
PUBLICATION_N = 4096
PROCESS_TIMEOUT_SECONDS = 1200

Results = dict[str, dict[str, float]]


@dataclass(frozen=True)
class ProcessRecord:
    before: PowerState
    after: PowerState
    results: Results

    def timings(self) -> dict[str, float]:
        return {kernel: row["ms"] for kernel, row in self.results.items()}

    def to_json(self) -> dict[str, object]:
        return {
            "before": self.before.to_json(),
            "after": self.after.to_json(),
            "kernels": self.results,
            "ratiosToNaive": ratios_to("naive", self.timings()),
            "ratiosToVendor": ratios_to("vendor", self.timings()),
        }


def spawn_process(n: int) -> Results:
    command = [sys.executable, "-m", "tiling_explainer.cli", "process", "--n", str(n)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=PROCESS_TIMEOUT_SECONDS)
    if done.returncode != 0:
        raise RuntimeError(f"measurement process failed:\n{done.stderr}")
    return json.loads(done.stdout.splitlines()[-1])


def print_record(index: int, n: int, record: ProcessRecord) -> None:
    to_naive = ratios_to("naive", record.timings())
    to_vendor = ratios_to("vendor", record.timings())
    print(f"process {index + 1}, n = {n}, before {record.before}, after {record.after}")
    for kernel, row in record.results.items():
        print(
            f"  {kernel:8} {row['ms']:9.3f} ms {gflops(n, row['ms']):7.0f} GFLOP/s"
            f"  x{to_naive[kernel]:5.2f} naive  {100 * to_vendor[kernel]:5.1f}% mx.matmul"
            f"  max abs err {row['maxAbsErr']:.2e}",
            flush=True,
        )


def measure(n: int, processes: int) -> list[ProcessRecord]:
    records = []
    for index in range(processes):
        before = read_power()
        results = spawn_process(n)
        record = ProcessRecord(before=before, after=read_power(), results=results)
        print_record(index, n, record)
        records.append(record)
    return records


def build_machine(
    chip: str,
    n: int,
    records: list[ProcessRecord],
    kernels: frozenset[str],
    publication: bool,
    versions: dict[str, str],
) -> Machine:
    states = [state for record in records for state in (record.before, record.after)]
    runs = [record.timings() for record in records]
    return {
        "id": machine_id(chip),
        "name": chip,
        "n": n,
        "runs": len(records),
        "conditions": {
            "acPower": all(state.ac_power for state in states),
            "lowPowerMode": any(state.low_power_mode for state in states),
            "publication": publication,
            "dtype": "float32",
            "timingMethod": "wall clock around mx.eval",
            "timingWarmups": WARMUPS,
            "timingLaunchesPerSample": BATCH,
            "timingSamples": SAMPLES,
        },
        "versions": versions,
        "kernels": [summarize(kernel, n, runs) for kernel in ORDER if kernel in kernels],
    }


def merged(path: Path, source: Source, machine: Machine) -> SpeedsDoc:
    machines = json.loads(path.read_text())["machines"] if path.exists() else []
    others = [entry for entry in machines if entry["id"] != machine["id"]]
    return {"version": VERSION, "source": source, "machines": [*others, machine]}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def write_outputs(
    folder: Path, n: int, records: list[ProcessRecord], publication: bool, commit: str
) -> None:
    source: Source = {"commit": commit, "code": code_url(commit)}
    chip, versions = chip_name(), {"commit": commit, **software_versions()}
    for name, kernels in (("speeds.json", SPEED_KERNELS), ("challenge.json", CHALLENGE_KERNELS)):
        machine = build_machine(chip, n, records, kernels, publication, versions)
        write_json(folder / name, merged(folder / name, source, machine))
    raw = {
        "version": VERSION,
        "source": source,
        "n": n,
        "publication": publication,
        "processes": [record.to_json() for record in records],
    }
    write_json(folder / "mac-runs.json", raw)


def run_mac(n: int, processes: int, allow_battery: bool) -> int:
    publication = n == PUBLICATION_N and not allow_battery
    commit = head_commit(ROOT)
    reason = refusal([read_power()], is_dirty(ROOT)) if publication else None
    if reason:
        print(f"refusing a publication run: {reason}. Pass --allow-battery for a test run.")
        return 2
    records = measure(n, processes)
    states = [state for record in records for state in (record.before, record.after)]
    reason = refusal(states, is_dirty(ROOT)) if publication else None
    if reason:
        print(f"conditions changed during the run ({reason}); nothing written to out/")
        write_outputs(RUNS, n, records, False, commit)
        return 2
    write_outputs(OUT if publication else RUNS, n, records, publication, commit)
    return 0
