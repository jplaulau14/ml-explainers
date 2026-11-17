import subprocess
from pathlib import Path

from tiling_explainer.a10.upload import (
    PACKAGE,
    code_problem,
    code_record,
    head_files,
    uploaded_files,
)


def git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    signing = ["-c", "commit.gpgsign=false"]
    subprocess.run(["git", *identity, *signing, *args], cwd=repo, check=True, capture_output=True)


def committed_package(tmp_path: Path) -> Path:
    package = tmp_path / PACKAGE
    (package / "a10").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "a10" / "run.py").write_text("print('v1')\n")
    (tmp_path / ".gitignore").write_text("local.py\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "code")
    return package


def problem(tmp_path: Path, package: Path) -> str | None:
    return code_problem(code_record(head_files(tmp_path)), code_record(uploaded_files(package)))


def test_clean_tree_uploads_exactly_head(tmp_path: Path) -> None:
    package = committed_package(tmp_path)
    assert code_record(head_files(tmp_path))["files"] == ["__init__.py", "a10/run.py"]
    assert problem(tmp_path, package) is None


def test_edited_file_differs_from_head(tmp_path: Path) -> None:
    package = committed_package(tmp_path)
    (package / "a10" / "run.py").write_text("print('v2')\n")
    assert problem(tmp_path, package) == "a call ran code that differs from HEAD"


def test_gitignored_file_under_the_package_is_an_extra_file(tmp_path: Path) -> None:
    package = committed_package(tmp_path)
    (package / "local.py").write_text("")
    assert problem(tmp_path, package) == (
        "a call ran a different file set than HEAD: extra ['local.py'], missing []"
    )
