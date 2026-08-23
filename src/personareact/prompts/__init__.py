"""Prompt loading, rendering and hashing.

Prompts are files, not string literals buried in modules. Three reasons:

1. they are the experimental object, so their exact text has to be citable and
   diffable;
2. their SHA-256 goes into every run manifest, so a result can always be traced
   back to the prompt that produced it;
3. domain-specific content lives under ``pt_BR/<domain>/``, which is what
   prevents a football Text-to-SQL prompt from being used on hospital records
   (see ``docs/ERRATA.md``, SV-06).

The locale directory (``pt_BR``) is explicit so that a future multilingual
replication is a new directory rather than a rewrite. The thesis flags the
Portuguese-only prompts as a limitation in section 5.6.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

from ..paths import PROMPTS_DIR

DEFAULT_LOCALE = "pt_BR"


class PromptError(RuntimeError):
    pass


def prompt_path(name: str, *, scope: str = "shared", locale: str = DEFAULT_LOCALE) -> Path:
    return PROMPTS_DIR / locale / scope / f"{name}.txt"


@lru_cache(maxsize=64)
def load_prompt(name: str, *, scope: str = "shared", locale: str = DEFAULT_LOCALE) -> str:
    path = prompt_path(name, scope=scope, locale=locale)
    if not path.is_file():
        raise PromptError(f"prompt not found: {path}")
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=64)
def prompt_hash(name: str, *, scope: str = "shared", locale: str = DEFAULT_LOCALE) -> str:
    text = load_prompt(name, scope=scope, locale=locale)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=16)
def load_profile_guidance(domain: str, *, locale: str = DEFAULT_LOCALE) -> dict[str, str]:
    """Per-profile tone guidance for a domain, as data."""
    path = PROMPTS_DIR / locale / domain / "profiles.yaml"
    if not path.is_file():
        raise PromptError(f"profile guidance not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = raw.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise PromptError(f"{path}: 'profiles' must be a mapping")
    return {str(k): str(v).strip() for k, v in profiles.items()}


def render_profile_guidance(domain: str, *, locale: str = DEFAULT_LOCALE) -> str:
    guidance = load_profile_guidance(domain, locale=locale)
    return "\n".join(f"- **{name}** → {text}" for name, text in guidance.items())


def render(
    name: str,
    *,
    scope: str = "shared",
    locale: str = DEFAULT_LOCALE,
    **values: object,
) -> str:
    template = load_prompt(name, scope=scope, locale=locale)
    try:
        return template.format(**values)
    except KeyError as exc:
        raise PromptError(
            f"prompt {scope}/{name}.txt needs placeholder {exc} which was not supplied"
        ) from None


def manifest_hashes(domain: str, *, locale: str = DEFAULT_LOCALE) -> dict[str, str]:
    """Every prompt that could influence a run in this domain."""
    hashes: dict[str, str] = {}
    for scope, names in (
        ("shared", ("reason", "tailor", "cot", "standard")),
        (domain, ("text_to_sql",)),
    ):
        for name in names:
            path = prompt_path(name, scope=scope, locale=locale)
            if path.is_file():
                hashes[f"{locale}/{scope}/{name}.txt"] = prompt_hash(
                    name, scope=scope, locale=locale
                )
    guidance = PROMPTS_DIR / locale / domain / "profiles.yaml"
    if guidance.is_file():
        hashes[f"{locale}/{domain}/profiles.yaml"] = hashlib.sha256(
            guidance.read_bytes()
        ).hexdigest()[:16]
    return hashes
