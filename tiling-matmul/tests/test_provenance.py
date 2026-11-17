import json
import subprocess
from pathlib import Path

import pytest

from tiling_explainer.driver import merged
from tiling_explainer.provenance import head_commit, stamp_problems

FOLDER = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    signing = ["-c", "commit.gpgsign=false"]
    subprocess.run(["git", *identity, *signing, *args], cwd=repo, check=True, capture_output=True)


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return head_commit(repo)


def write(folder: Path, name: str, payload: object) -> None:
    (folder / "out").mkdir(exist_ok=True)
    (folder / "out" / name).write_text(json.dumps(payload))


def source(commit: str) -> dict[str, str]:
    return {"commit": commit, "code": ""}


def machine(machine_id: str, commit: str) -> dict[str, object]:
    return {"id": machine_id, "versions": {"commit": commit}}


def machine_run(folder: Path, machine_id: str, sidecar: str, commit: str) -> None:
    speeds = folder / "out" / "speeds.json"
    write(folder, "speeds.json", merged(speeds, source(commit), machine(machine_id, commit)))
    write(folder, sidecar, {"source": source(commit)})


def repo_with_code(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    folder = tmp_path / "tiling-matmul"
    folder.mkdir()
    (folder / "kernels.py").write_text("v1\n")
    commit_all(tmp_path, "code")
    return folder


def mac_run_then_a10g_run(folder: Path) -> tuple[str, str]:
    mac = head_commit(folder)
    machine_run(folder, "m3-pro", "mac-runs.json", mac)
    commit_all(folder, "mac run")
    (folder / "README.md").write_text("results\n")
    a10g = commit_all(folder, "readme")
    machine_run(folder, "a10g", "a10g-runs.json", a10g)
    return mac, a10g


def test_committed_output_records_the_commits_that_made_it() -> None:
    if not list((FOLDER / "out").glob("*.json")):
        pytest.skip("no committed out/ files yet")
    assert stamp_problems(FOLDER) == []


def test_machines_run_and_committed_at_different_commits_pass(tmp_path: Path) -> None:
    folder = repo_with_code(tmp_path)
    mac, a10g = mac_run_then_a10g_run(folder)
    commit_all(tmp_path, "a10g run")
    doc = json.loads((folder / "out" / "speeds.json").read_text())
    assert doc["source"]["commit"] == a10g
    assert [m["versions"]["commit"] for m in doc["machines"]] == [mac, a10g]
    assert stamp_problems(folder) == []


def test_older_machine_entry_re_stamped_with_the_newer_commit_fails(tmp_path: Path) -> None:
    folder = repo_with_code(tmp_path)
    mac, a10g = mac_run_then_a10g_run(folder)
    speeds = folder / "out" / "speeds.json"
    doc = json.loads(speeds.read_text())
    doc["machines"][0]["versions"]["commit"] = a10g
    speeds.write_text(json.dumps(doc))
    commit_all(tmp_path, "a10g run")
    assert stamp_problems(folder) == [
        f"out/speeds.json: m3-pro commit {a10g} is not the run that last wrote out/mac-runs.json"
    ]


def test_rerun_that_edits_out_still_passes(tmp_path: Path) -> None:
    folder = repo_with_code(tmp_path)
    write(folder, "speeds.json", {"source": source(head_commit(folder))})
    commit_all(tmp_path, "first run")
    (folder / "kernels.py").write_text("v2\n")
    write(folder, "speeds.json", {"source": source(commit_all(tmp_path, "code fix"))})
    commit_all(tmp_path, "rerun")
    assert stamp_problems(folder) == []


def test_output_edited_after_a_code_change_without_a_rerun_fails(tmp_path: Path) -> None:
    folder = repo_with_code(tmp_path)
    first = head_commit(folder)
    write(folder, "speeds.json", {"source": source(first), "value": 1})
    commit_all(tmp_path, "first run")
    (folder / "kernels.py").write_text("v2\n")
    commit_all(tmp_path, "code fix")
    write(folder, "speeds.json", {"source": source(first), "value": 2})
    commit_all(tmp_path, "hand edit")
    assert stamp_problems(folder) == [
        f"out/speeds.json: source.commit {first} is not the commit that last wrote it"
    ]
