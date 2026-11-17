import ast
import io
import sys
import tokenize
from pathlib import Path

MAX_FILE_LINES = 400
MAX_FUNCTION_LINES = 40
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
DOCUMENTABLE = (ast.Module, ast.ClassDef, *FUNCTIONS)


def file_problems(source: str) -> list[str]:
    lines = len(source.splitlines())
    return [f"file has {lines} lines, limit {MAX_FILE_LINES}"] if lines > MAX_FILE_LINES else []


def function_problems(tree: ast.Module) -> list[str]:
    functions = [node for node in ast.walk(tree) if isinstance(node, FUNCTIONS)]
    spans = [(f, f.end_lineno - f.lineno + 1) for f in functions]
    return [
        f"line {f.lineno}: {f.name} has {span} lines, limit {MAX_FUNCTION_LINES}"
        for f, span in spans
        if span > MAX_FUNCTION_LINES
    ]


def comment_problems(source: str) -> list[str]:
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [f"line {t.start[0]}: comment" for t in tokens if t.type == tokenize.COMMENT]


def docstring_problems(tree: ast.Module) -> list[str]:
    nodes = [node for node in ast.walk(tree) if isinstance(node, DOCUMENTABLE)]
    documented = [node for node in nodes if ast.get_docstring(node) is not None]
    return [f"line {getattr(node, 'lineno', 1)}: docstring" for node in documented]


def problems(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source)
    found = [
        *file_problems(source),
        *function_problems(tree),
        *comment_problems(source),
        *docstring_problems(tree),
    ]
    return [f"{path}: {problem}" for problem in found]


def main(args: list[str]) -> int:
    paths = [Path(arg) for arg in args if arg.endswith(".py") and Path(arg).exists()]
    found = [problem for path in paths for problem in problems(path)]
    print("\n".join(found) or f"{len(paths)} files within limits")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
