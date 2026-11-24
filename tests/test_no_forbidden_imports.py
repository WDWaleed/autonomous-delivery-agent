"""Static AST scan ensuring no forbidden third-party solvers are imported in delivery/."""

import ast
from pathlib import Path

FORBIDDEN_MODULES = {
    "networkx",
    "simpleai",
    "aima",
    "gym",
    "gymnasium",
}

FORBIDDEN_FROM_SCIPY = {
    "scipy.sparse.csgraph",
    "csgraph",
}


def check_forbidden_imports(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                if name in FORBIDDEN_MODULES or alias.name in FORBIDDEN_FROM_SCIPY:
                    violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_root = node.module.split(".")[0]
                if mod_root in FORBIDDEN_MODULES or node.module in FORBIDDEN_FROM_SCIPY:
                    violations.append((node.lineno, node.module))
                for alias in node.names:
                    full_name = f"{node.module}.{alias.name}"
                    if full_name in FORBIDDEN_FROM_SCIPY:
                        violations.append((node.lineno, full_name))
    return violations


def test_no_forbidden_imports_in_delivery():
    project_root = Path(__file__).resolve().parent.parent
    delivery_dir = project_root / "delivery"

    all_violations = {}
    for py_file in delivery_dir.rglob("*.py"):
        v = check_forbidden_imports(py_file)
        if v:
            all_violations[str(py_file.relative_to(project_root))] = v

    assert not all_violations, f"Forbidden solver imports detected: {all_violations}"
