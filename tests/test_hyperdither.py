import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def third_party_imports():
    tree = ast.parse((ROOT / "hyperdither.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module.split(".")[0])
    return names - set(sys.stdlib_module_names)


def test_manifest_declares_every_third_party_import():
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.match(r"[A-Za-z0-9_.-]+", dep).group().lower()
        for dep in manifest["project"]["dependencies"]
    }
    wanted = {{"PIL": "pillow"}.get(name, name).lower() for name in third_party_imports()}
    assert wanted and wanted <= declared
