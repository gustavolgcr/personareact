"""Database builders, and the import path from the legacy tree.

The original had two standalone scripts that wrote their output into the
current working directory under one name (``fortaleza_trajetorias.db``,
``hsl_dados.db``) while every agent read a different name from a different
directory (``src/data/*.sqlite``). The rename between them was manual and
undocumented, and the result was nine copies of the football database spread
across nine project folders which then drifted into three distinct builds.

Here there is exactly one database per domain, at the path its config declares,
with a checksum recorded in that config and verified before every run.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from ..config import DomainConfig
from ..paths import DATA_DIR
from . import covid, soccer

BUILDERS: dict[str, Callable[..., Path]] = {
    "soccer": soccer.build,
    "covid": covid.build,
}

#: Where the legacy tree kept each already-built database.
LEGACY_DATABASES = {
    "soccer": "data-processing/src/data/fortaleza_trajetorias.sqlite",
    "covid": "data-processing/covid.sqlite",
}


def build_domain(
    domain: str,
    *,
    source: Path | None = None,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    config = DomainConfig.load(domain)
    target = config.database_path

    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists. Pass --force to rebuild, or "
            f"`personareact data verify` to check it."
        )

    builder = BUILDERS.get(domain)
    if builder is None:
        raise KeyError(f"no builder for domain {domain!r}; available: {', '.join(BUILDERS)}")

    resolved = source or (DATA_DIR / "source" / domain)
    return builder(resolved, target, on_progress=on_progress)


def import_legacy(
    domain: str,
    legacy_root: Path,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Copy a prebuilt database out of the original tree and fix its views.

    Views are metadata: rebuilding them costs milliseconds and touches no rows,
    so importing the existing football database and replacing its two broken
    views gives a correct database without re-reading 2.5M trajectory points.
    """
    config = DomainConfig.load(domain)
    relative = LEGACY_DATABASES.get(domain)
    if relative is None:
        raise KeyError(f"no legacy database mapping for {domain!r}")

    origin = legacy_root.expanduser().resolve() / relative
    if not origin.is_file():
        raise FileNotFoundError(f"legacy database not found: {origin}")

    target = config.database_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if on_progress:
        on_progress(f"copying {origin.name} ({origin.stat().st_size / 1e6:.0f} MB)")
    shutil.copyfile(origin, target)

    if domain == "soccer":
        soccer.rebuild_views(target, on_progress=on_progress)

    return target


__all__ = ["BUILDERS", "build_domain", "import_legacy", "covid", "soccer"]
