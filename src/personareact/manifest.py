"""Run manifests: what was run, with what, against what.

In the original implementation provenance had to be *inferred*. Establishing
that every answer file had been produced with the final prompt required
comparing git commit dates (prompts last changed 2025-07-27; answers committed
2025-08-08 and 2025-08-28) because nothing was recorded alongside the outputs.

A manifest removes the guesswork. It also carries the database checksum, which
would have caught the defect described in ``docs/ERRATA.md``: PersonaReAct and
the ReAct baselines queried *different builds* of the soccer database -- same
rows, but the baselines' copy had two pre-aggregated SQL views that
PersonaReAct's did not, and ``view_support=True`` put them in the schema handed
to the Text-to-SQL agent.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")


def new_run_id(prefix: str) -> str:
    return f"{utc_now()}-{prefix}"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


@dataclass
class Manifest:
    run_id: str
    kind: str
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_utc: str | None = None
    git_sha: str | None = field(default_factory=git_sha)
    git_dirty: bool | None = field(default_factory=git_dirty)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    package_version: str = "1.0.0"

    experiment: str | None = None
    config_fingerprint: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    database_hashes: dict[str, str] = field(default_factory=dict)
    corpus: dict[str, Any] = field(default_factory=dict)

    cells_total: int = 0
    cells_done: int = 0
    cells_skipped: int = 0
    cells_failed: int = 0
    answers_total: int = 0
    answers_failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, directory: Path) -> Manifest:
        path = directory / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"no manifest in {directory}")
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})
