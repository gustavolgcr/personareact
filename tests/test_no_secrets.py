"""No credential may enter the repository.

This exists because of where the project came from: the original tree carried
twelve copies of a live OpenAI key in twelve `.env` files. They were correctly
gitignored, but twelve copies of a credential are one `git add -f` away from
being permanent, and GitHub's push protection only covers public repositories.

A test costs nothing and runs everywhere, including in a private repo.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from personareact.paths import PROJECT_ROOT

#: Deliberately narrow: long, high-entropy tails only, so a placeholder like
#: `sk-proj-replace-me` in .env.example does not trip it.
SECRET_PATTERNS = [
    re.compile(r"sk-proj-[A-Za-z0-9_-]{40,}"),
    re.compile(r"sk-[A-Za-z0-9]{40,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

SKIP_SUFFIXES = (".png", ".pdf", ".sqlite", ".db", ".jsonl")


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line]


@pytest.mark.skipif(not tracked_files(), reason="not a git checkout")
def test_no_tracked_file_contains_a_credential():
    offenders: list[str] = []
    for name in tracked_files():
        if name.endswith(SKIP_SUFFIXES):
            continue
        path = PROJECT_ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{name} matches {pattern.pattern}")
    assert not offenders, "credentials found in tracked files:\n" + "\n".join(offenders)


@pytest.mark.skipif(not tracked_files(), reason="not a git checkout")
def test_the_env_file_is_ignored():
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "check-ignore", ".env"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, ".env is not gitignored"


def test_the_example_env_holds_only_a_placeholder():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "replace-me" in text
    for pattern in SECRET_PATTERNS:
        assert not pattern.search(text)
