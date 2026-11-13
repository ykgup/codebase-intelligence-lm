"""
Build structured repository context (file tree, imports, classes, functions)
using Python's AST module to evaluate repository-level code understanding.
"""

import ast
from pathlib import Path


IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "checkpoints",
    "raw",
    "processed",
    ".idea",
    ".vscode",
}


def analyze_python_file(file_path: Path):
    """Extract module docstring, imports, classes, and functions via AST."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    imports = []
    classes = []
    functions = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(alias.name for alias in node.names)
            imports.append(f"{module} ({names})")
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body if isinstance(n, ast.FunctionDef)
            ]
            classes.append(f"{node.name} [methods: {', '.join(methods)}]")
        elif isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            functions.append(f"{node.name}({', '.join(args)})")

    return {
        "docstring": ast.get_docstring(tree),
        "imports": imports,
        "classes": classes,
        "functions": functions,
    }


def build_repository_context(repo_root: Path) -> str:
    """Generate a structured markdown summary of a Python repository."""
    repo_root = repo_root.resolve()
    py_files = []

    for path in sorted(repo_root.rglob("*.py")):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        py_files.append(path)

    lines = [
        f"# Structured Repository Context: {repo_root.name}",
        "",
        "## 1. Repository Python Files",
    ]

    for path in py_files:
        rel_path = path.relative_to(repo_root).as_posix()
        lines.append(f"- `{rel_path}`")

    lines.append("\n## 2. Module Structure & Signatures")

    for path in py_files:
        rel_path = path.relative_to(repo_root).as_posix()
        info = analyze_python_file(path)
        lines.append(f"\n### File: `{rel_path}`")

        if info is None:
            lines.append("- Could not parse AST.")
            continue

        if info["docstring"]:
            first_line = info["docstring"].strip().splitlines()[0]
            lines.append(f"- **Purpose:** {first_line}")

        if info["imports"]:
            lines.append(f"- **Imports:** {', '.join(info['imports'])}")

        if info["classes"]:
            lines.append(f"- **Classes:** {', '.join(info['classes'])}")

        if info["functions"]:
            lines.append(f"- **Functions:** {', '.join(info['functions'])}")

    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parent.parent
    context = build_repository_context(root)
    print(context)


if __name__ == "__main__":
    main()