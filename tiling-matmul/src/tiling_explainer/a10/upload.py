import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

PACKAGE = "src/tiling_explainer"
GIT_TIMEOUT_SECONDS = 30


class CodeRecord(TypedDict):
    files: list[str]
    digest: str


def code_record(files: Mapping[str, bytes]) -> CodeRecord:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(f"{path}\0{hashlib.sha256(files[path]).hexdigest()}\n".encode())
    return {"files": sorted(files), "digest": digest.hexdigest()}


def uploaded_files(package: Path) -> dict[str, bytes]:
    return {
        path.relative_to(package).as_posix(): path.read_bytes()
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def git_bytes(root: Path, *args: str) -> bytes:
    done = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, timeout=GIT_TIMEOUT_SECONDS
    )
    return done.stdout


def head_files(root: Path) -> dict[str, bytes]:
    listed = git_bytes(root, "ls-tree", "-r", "--name-only", "HEAD", "--", PACKAGE)
    paths = [path for path in listed.decode().splitlines() if path.endswith(".py")]
    return {
        Path(path).relative_to(PACKAGE).as_posix(): git_bytes(root, "show", f"HEAD:./{path}")
        for path in paths
    }


def code_problem(expected: CodeRecord, found: object) -> str | None:
    if found == expected:
        return None
    files = found.get("files") if isinstance(found, dict) else None
    if not isinstance(files, list):
        return "a call did not report the code it ran"
    extra = sorted(set(map(str, files)) - set(expected["files"]))
    missing = sorted(set(expected["files"]) - set(map(str, files)))
    if extra or missing:
        return f"a call ran a different file set than HEAD: extra {extra}, missing {missing}"
    return "a call ran code that differs from HEAD"
