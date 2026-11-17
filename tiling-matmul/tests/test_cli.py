import pytest

from tiling_explainer import cli
from tiling_explainer.a10 import run


def test_more_than_eight_timing_calls_is_refused_before_any_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[tuple[int, int]] = []
    monkeypatch.setattr(run, "run_a10", lambda n, calls: started.append((n, calls)) or 0)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["a10", "--calls", "9"])
    assert (stopped.value.code, started) == ("--calls must be from 3 to 8", [])


def test_eight_timing_calls_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple[int, int]] = []
    monkeypatch.setattr(run, "run_a10", lambda n, calls: started.append((n, calls)) or 0)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["a10", "--calls", "8"])
    assert (stopped.value.code, started) == (0, [(4096, 8)])
