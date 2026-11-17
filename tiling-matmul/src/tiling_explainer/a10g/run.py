import subprocess

from tiling_explainer.a10g.app import run_calls
from tiling_explainer.a10g.ncu import ProfileError, measured_launches, parse_launches
from tiling_explainer.a10g.outputs import run_problem, write_a10g
from tiling_explainer.contract import vendor_kernel_problem
from tiling_explainer.driver import OUT, PUBLICATION_N, ROOT, RUNS, write_json
from tiling_explainer.provenance import head_commit, is_dirty

LOCK_CHECK_TIMEOUT_SECONDS = 120


def lock_is_current() -> bool:
    done = subprocess.run(
        ["uv", "lock", "--check"], cwd=ROOT, capture_output=True, timeout=LOCK_CHECK_TIMEOUT_SECONDS
    )
    return done.returncode == 0


def refusal(publication: bool, dirty: bool, lock_current: bool) -> str | None:
    if publication and dirty:
        return "uncommitted changes in tiling-matmul/ outside out/"
    if not lock_current:
        return "uv.lock does not match pyproject.toml; run uv lock and commit it"
    return None


def profile_problem(launches: list[dict[str, str | float]]) -> str | None:
    try:
        measured = measured_launches(launches)
    except ProfileError as error:
        return str(error)
    return vendor_kernel_problem(measured["vendor"]["name"])


def run_a10g(n: int, calls: int) -> int:
    publication = n == PUBLICATION_N
    commit = head_commit(ROOT)
    reason = refusal(publication, is_dirty(ROOT), lock_is_current())
    if reason:
        print(f"refusing to start the A10G run: {reason}")
        return 2
    timings, profile = run_calls(n, calls, commit)
    write_json(RUNS / "a10g-calls.json", {"calls": timings, "profile": profile})
    launches = parse_launches(str(profile["csv"]))
    problem = run_problem(timings, profile, commit) or profile_problem(launches)
    if problem:
        print(f"{problem}; the raw calls are in runs/a10g-calls.json and nothing went to out/")
        return 2
    folder = OUT if publication else RUNS
    write_a10g(folder, n, timings, profile, launches, commit, publication)
    return 0
