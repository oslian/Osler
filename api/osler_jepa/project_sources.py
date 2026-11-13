"""The project source files loaded in this process and the interpreter and package versions: the inventory a frozen
protocol hashes and an extraction report binds.

A module named like a project module but loaded from elsewhere, or a project file loaded from anything but its
``.py`` source, raises.  Relative pseudo-paths (torch.ops, torch.classes) are not project files.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from osler_jepa.frozen_protocol import sha256

PROJECT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "torch")


def environment() -> dict:
    """Interpreter and numerical package versions."""

    return {"python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES}}


def project_module_name(name: str) -> bool:
    top = name.split(".")[0]
    return (PROJECT / f"{top}.py").exists() or (PROJECT / top / "__init__.py").exists()


def loaded_project_files() -> set[str]:
    """Project source files loaded in this process, relative to the project (the protocol's path convention)."""

    files = set()
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file)
        inside = path.is_absolute() and path.resolve().is_relative_to(PROJECT) and ".venv" not in path.resolve().parts
        if project_module_name(name) and not inside:
            raise RuntimeError(f"project module {name} is not loaded from {PROJECT}: {file}")
        if inside:
            if path.suffix != ".py":
                raise RuntimeError(f"project module {name} is not loaded from its source: {file}")
            files.add(str(path.resolve().relative_to(PROJECT)))
    return files


def source_digests() -> dict[str, str]:
    """SHA-256 of every project source file loaded in this process."""

    return {name: sha256(PROJECT / name) for name in sorted(loaded_project_files())}


def digest_of(files: dict[str, str]) -> str:
    """One digest of a {file: sha256} inventory."""

    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
