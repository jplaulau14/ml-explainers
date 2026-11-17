from pathlib import Path

import pytest
from a10_fakes import COMMIT, FILES, call, profile

from tiling_explainer import driver
from tiling_explainer.a10 import run

OUTPUTS = ["a10-runs.json", "bytes.json", "challenge.json", "speeds.json"]


def fake_a10(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dirty: bool = False,
    lock: bool = True,
    calls: list[dict[str, object]] | None = None,
    run_profile: dict[str, object] | None = None,
) -> list[int]:
    started: list[int] = []

    def run_calls(n: int, count: int, commit: str) -> tuple[list, dict, list]:
        started.append(n)
        return calls or [call(0), call(1), call(2)], run_profile or profile(), []

    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(run, "OUT", tmp_path / "out")
    monkeypatch.setattr(run, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(run, "head_commit", lambda folder: COMMIT)
    monkeypatch.setattr(run, "is_dirty", lambda folder: dirty)
    monkeypatch.setattr(run, "lock_is_current", lambda: lock)
    monkeypatch.setattr(run, "head_files", lambda folder: FILES)
    monkeypatch.setattr(run, "run_calls", run_calls)
    return started


def names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


@pytest.mark.parametrize("n", [4096, 1024])
def test_dirty_tree_refuses_before_any_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], n: int
) -> None:
    started = fake_a10(monkeypatch, tmp_path, dirty=True)
    assert run.run_a10(n, 3) == 2
    assert "uncommitted changes" in capsys.readouterr().out
    assert (started, list(tmp_path.iterdir())) == ([], [])


def test_stale_lock_refuses_before_any_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    started = fake_a10(monkeypatch, tmp_path, lock=False)
    assert run.run_a10(1024, 3) == 2
    assert "uv.lock does not match" in capsys.readouterr().out
    assert (started, list(tmp_path.iterdir())) == ([], [])


def test_vendor_kernel_with_tensor_activity_leaves_out_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_a10(monkeypatch, tmp_path, run_profile=profile(cycles=512))
    assert run.run_a10(4096, 3) == 2
    assert "used the tensor pipe" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
    assert names(tmp_path / "runs") == ["a10-calls.json"]


def test_outputs_that_fail_the_contract_leave_out_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = [call(0, vendor_ms=-9.0), call(1, vendor_ms=-9.0), call(2, vendor_ms=-9.0)]
    fake_a10(monkeypatch, tmp_path, calls=calls)
    assert run.run_a10(4096, 3) == 2
    assert "speeds.json: machines[0].vendor: ratios" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_smaller_n_writes_only_to_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_a10(monkeypatch, tmp_path)
    assert run.run_a10(1024, 3) == 0
    assert not (tmp_path / "out").exists()
    assert names(tmp_path / "runs") == sorted(["a10-calls.json", *OUTPUTS])


def test_clean_publication_run_writes_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_a10(monkeypatch, tmp_path)
    assert run.run_a10(4096, 3) == 0
    assert names(tmp_path / "out") == OUTPUTS
