"""The app must recover when a code reload leaves the rag package half-loaded."""
import ast
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def _loader():
    src = APP.read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and
            all(a.name not in ("streamlit", "pandas") for a in n.names)]
    keep += [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_RAG"]
    keep += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_rag"]
    ns = {}
    exec(compile(ast.Module(keep, []), str(APP), "exec"), ns)
    return ns["_load_rag"]


def test_recovers_from_missing_submodule():
    load = _loader()
    load()
    sys.modules["rag.search"] = None            # what a half-finished reload looks like
    sys.modules.pop("rag.answer", None)
    mods = load()
    assert [m.__name__ for m in mods][:2] == ["rag.answer", "rag.compliance"]
    assert sys.modules["rag.search"] is not None
