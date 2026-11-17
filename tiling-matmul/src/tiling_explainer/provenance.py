import json
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/jplaulau14/ml-explainers"
SIDECARS = {"m3-pro": "out/mac-runs.json", "a10": "out/a10-runs.json"}


def git(folder: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=folder, check=True, capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip()


def head_commit(folder: Path) -> str:
    return git(folder, "rev-parse", "HEAD")


def code_url(commit: str, folder: str = "tiling-matmul") -> str:
    return f"{REPO_URL}/tree/{commit}/{folder}"


def is_dirty(folder: Path) -> bool:
    return bool(git(folder, "status", "--porcelain", "--", ".", ":(exclude)out"))


def allowed_commits(folder: Path, path: str) -> set[str]:
    last = git(folder, "log", "-1", "--format=%H", "--", path)
    if not last:
        return set()
    parents = git(folder, "rev-parse", f"{last}^@").splitlines()
    return {last, *parents[:1]}


def machine_problems(folder: Path, name: str, machine: dict[str, object]) -> list[str]:
    machine_id = str(machine.get("id"))
    sidecar = SIDECARS.get(machine_id)
    versions = machine.get("versions")
    commit = versions.get("commit") if isinstance(versions, dict) else None
    if sidecar is None or commit not in allowed_commits(folder, sidecar):
        return [f"{name}: {machine_id} commit {commit} is not the run that last wrote {sidecar}"]
    return []


def stamp_problems(folder: Path) -> list[str]:
    found = []
    for path in sorted((folder / "out").glob("*.json")):
        name = f"out/{path.name}"
        doc = json.loads(path.read_text())
        commit = doc["source"]["commit"]
        if commit not in allowed_commits(folder, name):
            found.append(f"{name}: source.commit {commit} is not the commit that last wrote it")
        for machine in doc.get("machines", []):
            found += machine_problems(folder, name, machine)
    return found
