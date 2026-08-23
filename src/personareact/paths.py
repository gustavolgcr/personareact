"""Filesystem layout.

Every path in the project is derived from ``PROJECT_ROOT``. Nothing anywhere in
the codebase is allowed to hardcode an absolute path -- that was the single
biggest portability defect in the original implementation (the analysis
notebooks embedded ``/Users/<name>/...`` in every cell, so no reader could
execute them).

Resolution order for the project root:

1. ``$PERSONAREACT_ROOT`` if set (useful when the package is installed
   elsewhere but the data and configs live in a checkout);
2. the directory containing ``pyproject.toml``, walking up from this file.
"""

from __future__ import annotations

import os
from pathlib import Path


def _discover_root() -> Path:
    env = os.environ.get("PERSONAREACT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Installed without a checkout: fall back to the current directory.
    return Path.cwd().resolve()


PROJECT_ROOT: Path = _discover_root()

CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
TABLES_DIR: Path = PROJECT_ROOT / "tables"

PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROMPTS_DIR: Path = PACKAGE_DIR / "prompts"


def run_dir(run_id: str) -> Path:
    return RESULTS_DIR / run_id


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
