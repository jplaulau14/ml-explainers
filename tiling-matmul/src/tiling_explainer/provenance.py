import subprocess
from pathlib import Path

REPO_URL = "https://github.com/jplaulau14/ml-explainers"


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


def allowed_commits(folder: Path, out: str = "out") -> set[str]:
    last = git(folder, "log", "-1", "--format=%H", "--", out)
    if not last:
        return set()
    parents = git(folder, "rev-parse", f"{last}^@").splitlines()
    return {last, *parents[:1]}
