import json
import subprocess
from pathlib import Path

import pytest

from tiling_explainer.provenance import allowed_commits, head_commit

FOLDER = Path(__file__).resolve().parents[1]
STAMPED = sorted(path for path in (FOLDER / "out").glob("*.json"))


def git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True)


def stamp(folder: Path, commit: str, value: float) -> None:
    out = folder / "out"
    out.mkdir(exist_ok=True)
    payload = {"source": {"commit": commit, "code": ""}, "value": value}
    (out / "speeds.json").write_text(json.dumps(payload))


def commit_all(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)


def recorded(folder: Path) -> str:
    return json.loads((folder / "out" / "speeds.json").read_text())["source"]["commit"]


@pytest.mark.parametrize("path", STAMPED, ids=lambda p: p.name)
def test_committed_output_records_the_commit_that_made_it(path: Path) -> None:
    assert json.loads(path.read_text())["source"]["commit"] in allowed_commits(FOLDER)


def test_rerun_that_edits_out_still_passes(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    folder = tmp_path / "tiling-matmul"
    folder.mkdir()
    (folder / "kernels.py").write_text("v1\n")
    commit_all(tmp_path, "code")
    stamp(folder, head_commit(folder), 1.0)
    commit_all(tmp_path, "first run")
    assert recorded(folder) in allowed_commits(folder)
    (folder / "kernels.py").write_text("v2\n")
    commit_all(tmp_path, "code fix")
    stamp(folder, head_commit(folder), 2.0)
    commit_all(tmp_path, "rerun")
    assert recorded(folder) in allowed_commits(folder)


def test_output_edited_after_a_code_change_without_a_rerun_fails(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "kernels.py").write_text("v1\n")
    commit_all(tmp_path, "code")
    stamp(tmp_path, head_commit(tmp_path), 1.0)
    commit_all(tmp_path, "first run")
    (tmp_path / "kernels.py").write_text("v2\n")
    commit_all(tmp_path, "code fix")
    stamp(tmp_path, recorded(tmp_path), 2.0)
    commit_all(tmp_path, "hand edit")
    assert recorded(tmp_path) not in allowed_commits(tmp_path)
