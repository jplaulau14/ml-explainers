import subprocess

from tiling_explainer.a10.app import run_calls
from tiling_explainer.a10.ncu import ProfileError, measured_launches, parse_launches
from tiling_explainer.a10.outputs import a10_documents, document_problems, run_problem, tensor_names
from tiling_explainer.a10.upload import code_record, head_files
from tiling_explainer.contract import PUBLICATION_N, vendor_kernel_problem
from tiling_explainer.driver import OUT, ROOT, RUNS, write_json
from tiling_explainer.provenance import head_commit, is_dirty

LOCK_CHECK_TIMEOUT_SECONDS = 120
NOTHING_WRITTEN = "the raw calls are in runs/a10-calls.json and nothing went to out/"


def lock_is_current() -> bool:
    done = subprocess.run(
        ["uv", "lock", "--check"], cwd=ROOT, capture_output=True, timeout=LOCK_CHECK_TIMEOUT_SECONDS
    )
    return done.returncode == 0


def refusal(dirty: bool, lock_current: bool) -> str | None:
    if dirty:
        return "uncommitted changes in tiling-matmul/ outside out/"
    if not lock_current:
        return "uv.lock does not match pyproject.toml; run uv lock and commit it"
    return None


def profile_problem(launches: list[dict[str, str | float]], tensors: list[str]) -> str | None:
    try:
        measured = measured_launches(launches)
    except ProfileError as error:
        return str(error)
    vendor = measured["vendor"]
    return vendor_kernel_problem(vendor["name"], {name: vendor.get(name) for name in tensors})


def run_a10(n: int, calls: int) -> int:
    publication = n == PUBLICATION_N
    commit = head_commit(ROOT)
    reason = refusal(is_dirty(ROOT), lock_is_current())
    if reason:
        print(f"refusing to start the A10 run: {reason}")
        return 2
    code = code_record(head_files(ROOT))
    timings, profile, refusals = run_calls(n, calls, commit)
    write_json(
        RUNS / "a10-calls.json", {"calls": timings, "profile": profile, "refusals": refusals}
    )
    launches = parse_launches(str(profile["csv"]))
    tensors = tensor_names(profile)
    problem = run_problem(timings, profile, code) or profile_problem(launches, tensors)
    if problem:
        print(f"{problem}; {NOTHING_WRITTEN}")
        return 2
    folder = OUT if publication else RUNS
    documents = a10_documents(folder, n, timings, profile, refusals, launches, commit, publication)
    found = document_problems(documents) if publication else []
    if found:
        print(f"{'; '.join(found)}; {NOTHING_WRITTEN}")
        return 2
    for path, document in documents.items():
        write_json(path, document)
    return 0
