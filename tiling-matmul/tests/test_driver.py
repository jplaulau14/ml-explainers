import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from tiling_explainer import driver
from tiling_explainer.conditions import PowerState

AC = PowerState(ac_power=True, low_power_mode=False)
BATTERY = PowerState(ac_power=False, low_power_mode=False)
TIMES = {"naive": 200.0, "tile32": 190.0, "reg4x4": 80.0, "reg8x8": 280.0, "vendor": 40.0}


def fake_mac(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, power: Iterator[PowerState]) -> None:
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "OUT", tmp_path / "out")
    monkeypatch.setattr(driver, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(driver, "read_power", lambda: next(power))
    monkeypatch.setattr(driver, "is_dirty", lambda folder: False)
    monkeypatch.setattr(driver, "head_commit", lambda folder: "c" * 40)
    monkeypatch.setattr(driver, "chip_name", lambda: "Apple M3 Pro")
    results = {kernel: {"ms": ms, "maxAbsErr": 0.0} for kernel, ms in TIMES.items()}
    monkeypatch.setattr(driver, "spawn_process", lambda n: results)


def test_battery_reading_mid_run_writes_runs_and_leaves_out_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_mac(monkeypatch, tmp_path, iter([AC, AC, AC, AC, BATTERY, AC, AC]))
    assert driver.run_mac(4096, 3, allow_battery=False) == 2
    assert not (tmp_path / "out").exists()
    speeds = json.loads((tmp_path / "runs" / "speeds.json").read_text())
    assert speeds["machines"][0]["conditions"]["publication"] is False
    assert speeds["machines"][0]["conditions"]["acPower"] is False


def test_battery_at_start_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_mac(monkeypatch, tmp_path, iter([BATTERY]))
    assert driver.run_mac(4096, 3, allow_battery=False) == 2
    assert list(tmp_path.iterdir()) == []
