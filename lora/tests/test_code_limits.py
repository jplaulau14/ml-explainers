import ast
import io
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_LINES = 400
MAX_FUNCTION_LINES = 40
SOURCES = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py")])
DOCUMENTABLE = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


@pytest.mark.parametrize("path", SOURCES, ids=relative)
def test_file_length(path: Path) -> None:
    assert len(path.read_text().splitlines()) <= MAX_FILE_LINES


@pytest.mark.parametrize("path", SOURCES, ids=relative)
def test_function_length(path: Path) -> None:
    tree = ast.parse(path.read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    too_long = [f.name for f in functions if f.end_lineno - f.lineno + 1 > MAX_FUNCTION_LINES]
    assert too_long == []


@pytest.mark.parametrize("path", SOURCES, ids=relative)
def test_no_comments(path: Path) -> None:
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
    comments = [t.start[0] for t in tokens if t.type == tokenize.COMMENT]
    assert comments == []


@pytest.mark.parametrize("path", SOURCES, ids=relative)
def test_no_docstrings(path: Path) -> None:
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, DOCUMENTABLE)]
    assert [n for n in nodes if ast.get_docstring(n) is not None] == []
