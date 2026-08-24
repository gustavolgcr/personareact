"""Configuration objects, loaded from ``configs/*.yaml``.

Everything an experiment varies is a config file, never a source edit. In the
original implementation the domain was selected by commenting out one of two
``DB_PATH`` lines, the model by editing a string literal, and the profile by
commenting out five of six list entries -- which meant no run was reproducible
and the on-disk state matched no reported result.

Four config kinds compose into a single :class:`CellConfig`, the unit of work:

    domain x approach x model x judge  ->  cell

A :class:`ExperimentConfig` is a declarative sweep over those axes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR, DATA_DIR, PROJECT_ROOT


class ConfigError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        available = sorted(p.stem for p in path.parent.glob("*.yaml"))
        raise ConfigError(
            f"config not found: {path.relative_to(PROJECT_ROOT)}\n"
            f"available in {path.parent.name}/: {', '.join(available) or '(none)'}"
        )
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


# --------------------------------------------------------------------------- #
# Domain
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DomainConfig:
    """A case study: its database, its profiles, its prompts.

    ``prompt_dir`` is domain-scoped on purpose. In the original code a single
    ``prompts.py`` served both case studies, so the COVID runs used a
    Text-to-SQL prompt whose closing instruction reads "A player can have
    multiple segments in the same match ... calculate the AVERAGE", and a
    personalization prompt whose worked examples were four football personas.
    """

    name: str
    display_name: str
    database: str
    database_sha256: str | None
    expects_views: list[str]
    profiles: list[str]
    prompt_dir: str
    description: str = ""

    @property
    def database_path(self) -> Path:
        return DATA_DIR / self.database

    @classmethod
    def load(cls, name: str) -> DomainConfig:
        raw = _load_yaml(CONFIG_DIR / "domain" / f"{name}.yaml")
        return cls(
            name=raw.get("name", name),
            display_name=raw.get("display_name", name),
            database=raw["database"],
            database_sha256=raw.get("database_sha256"),
            expects_views=list(raw.get("expects_views") or []),
            profiles=list(raw["profiles"]),
            prompt_dir=raw.get("prompt_dir", name),
            description=raw.get("description", ""),
        )


# --------------------------------------------------------------------------- #
# Approach
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ApproachConfig:
    """One of the six interaction patterns.

    The whole family is expressed as flags over a single graph rather than as
    six copy-pasted projects:

    ==================  =========  =====  ======  ==============================
    approach            reasoning  tools  tailor  profile reaches the model via
    ==================  =========  =====  ======  ==============================
    standard            no         no     no      nothing
    cot                 yes        no     no      nothing
    act                 no         yes    no      nothing
    react               yes        yes    no      nothing
    react_enriched      yes        yes    no      suffix appended to the question
    personareact        yes        yes    yes     the Tailor step
    ==================  =========  =====  ======  ==============================
    """

    name: str
    display_name: str
    use_reasoning: bool
    use_tools: bool
    use_tailor: bool
    max_iterations: int = 5
    question_suffix: str | None = None
    #: What the Tailor step is shown: "context" (the whole scratchpad, which is
    #: what the thesis implementation did) or "observations" (only O*, which is
    #: what section 4.2 formalises). See AgentState.tailor_input.
    tailor_sees: str = "context"
    description: str = ""

    @property
    def uses_profile(self) -> bool:
        return self.use_tailor or self.question_suffix is not None

    @classmethod
    def load(cls, name: str) -> ApproachConfig:
        raw = _load_yaml(CONFIG_DIR / "approach" / f"{name}.yaml")
        return cls(
            name=raw.get("name", name),
            display_name=raw.get("display_name", name),
            use_reasoning=bool(raw["use_reasoning"]),
            use_tools=bool(raw["use_tools"]),
            use_tailor=bool(raw["use_tailor"]),
            max_iterations=int(raw.get("max_iterations", 5)),
            question_suffix=raw.get("question_suffix"),
            tailor_sees=raw.get("tailor_sees", "context"),
            description=raw.get("description", ""),
        )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """A pinned model snapshot plus its sampling parameters.

    ``sql_model`` is deliberately separate from ``model``. In the original code
    the Text-to-SQL tool instantiated its own ``ChatOpenAI`` with the same
    global ``OPENAI_MODEL``, so sweeping the model swept the SQL generator too
    and the cross-model comparison conflated personalization quality with
    text-to-SQL quality. Holding ``sql_model`` fixed isolates the axis under
    test; leaving it null reproduces the original coupled behaviour.
    """

    name: str
    model: str
    sql_model: str | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    frequency_penalty: float | None = 0.0
    seed: int | None = 20250820
    reasoning: bool = False
    supports_sampling_params: bool = True
    usd_per_1m_input: float = 0.0
    usd_per_1m_output: float = 0.0

    @property
    def effective_sql_model(self) -> str:
        return self.sql_model or self.model

    def sampling_kwargs(self) -> dict[str, Any]:
        """Parameters actually sent to the API.

        Reasoning models (o-series, GPT-5) reject non-default sampling
        parameters, so those configs set ``supports_sampling_params: false``
        and we send nothing. The thesis quotes ``top_p = 1.0`` and
        ``frequency_penalty = 0.0`` as if they had been configured; they never
        were, and they are API defaults. Here they are explicit.
        """
        if not self.supports_sampling_params:
            return {}
        # Passed as first-class ChatOpenAI fields, not through `model_kwargs`,
        # which langchain-openai warns about and which silently bypasses its
        # own validation.
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    @classmethod
    def load(cls, name: str) -> ModelConfig:
        raw = _load_yaml(CONFIG_DIR / "model" / f"{name}.yaml")
        return cls(
            name=raw.get("name", name),
            model=raw["model"],
            sql_model=raw.get("sql_model"),
            temperature=raw.get("temperature", 1.0),
            top_p=raw.get("top_p", 1.0),
            frequency_penalty=raw.get("frequency_penalty", 0.0),
            seed=raw.get("seed", 20250820),
            reasoning=bool(raw.get("reasoning", False)),
            supports_sampling_params=bool(raw.get("supports_sampling_params", True)),
            usd_per_1m_input=float(raw.get("usd_per_1m_input", 0.0)),
            usd_per_1m_output=float(raw.get("usd_per_1m_output", 0.0)),
        )


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JudgeConfig:
    """LLM-as-a-judge settings.

    ``model`` must be a dated snapshot. The original code left
    ``# model=OPENAI_MODEL`` commented out on every metric, so deepeval fell
    back to its default (``gpt-4o``) -- an alias that OpenAI repoints over
    time, which makes the reported scores unrecoverable later.
    """

    name: str
    model: str
    repetitions: int = 5
    blind: bool = True
    criteria: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, name: str) -> JudgeConfig:
        raw = _load_yaml(CONFIG_DIR / "judge" / f"{name}.yaml")
        model = raw["model"]
        if model in {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"}:
            raise ConfigError(
                f"judge model {model!r} is a floating alias. Pin a dated snapshot "
                f"(e.g. gpt-4o-2024-11-20) so the scores stay recoverable."
            )
        return cls(
            name=raw.get("name", name),
            model=model,
            repetitions=int(raw.get("repetitions", 5)),
            blind=bool(raw.get("blind", True)),
            criteria=list(raw.get("criteria") or []),
        )


# --------------------------------------------------------------------------- #
# Cell + experiment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellConfig:
    """One (domain, approach, model, profile, difficulty) unit of work."""

    domain: DomainConfig
    approach: ApproachConfig
    model: ModelConfig
    profile: str
    difficulty: str

    @property
    def slug(self) -> str:
        return "__".join(
            [
                self.domain.name,
                self.approach.name,
                self.model.name,
                slugify(self.profile),
                self.difficulty,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.name,
            "approach": self.approach.name,
            "model": self.model.name,
            "model_snapshot": self.model.model,
            "sql_model_snapshot": self.model.effective_sql_model,
            "profile": self.profile,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    domains: list[str]
    approaches: list[str]
    models: list[str]
    difficulties: list[str]
    judge: str
    profiles: list[str] | None = None
    limit_questions: int | None = None

    @classmethod
    def load(cls, name: str) -> ExperimentConfig:
        raw = _load_yaml(CONFIG_DIR / "experiment" / f"{name}.yaml")
        return cls(
            name=raw.get("name", name),
            description=raw.get("description", ""),
            domains=list(raw["domains"]),
            approaches=list(raw["approaches"]),
            models=list(raw["models"]),
            difficulties=list(raw.get("difficulties") or ["easy", "hard"]),
            judge=raw.get("judge", "geval-gpt4o-2024-11-20"),
            profiles=list(raw["profiles"]) if raw.get("profiles") else None,
            limit_questions=raw.get("limit_questions"),
        )

    def cells(self) -> list[CellConfig]:
        """Expand the sweep.

        Every approach runs on every profile, including the profile-blind ones
        (``standard``, ``cot``, ``act``, ``react``). Those approaches never see
        the profile string, but they answer *the profile's* question, so each
        condition is comparable cell by cell.

        The original code did the opposite: plain ReAct was run once per
        (difficulty, model) against the *neutral* questions, and the evaluator
        then re-used that single answer set for all six profiles while telling
        the judge it had answered the profile-adapted question.
        """
        out: list[CellConfig] = []
        for domain_name in self.domains:
            domain = DomainConfig.load(domain_name)
            profiles = self.profiles or domain.profiles
            for approach_name in self.approaches:
                approach = ApproachConfig.load(approach_name)
                for model_name in self.models:
                    model = ModelConfig.load(model_name)
                    for profile in profiles:
                        for difficulty in self.difficulties:
                            out.append(
                                CellConfig(
                                    domain=domain,
                                    approach=approach,
                                    model=model,
                                    profile=profile,
                                    difficulty=difficulty,
                                )
                            )
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def slugify(text: str) -> str:
    """ASCII, lowercase, underscore-separated.

    Kept byte-compatible with the original ``normalizar_string`` so legacy
    filenames (``fisioterapeuta_do_time``) still resolve.
    """
    import unicodedata

    stripped = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return stripped.lower().replace(" ", "_")


def config_fingerprint(*objects: Any) -> str:
    """Stable hash of the resolved configuration, recorded in the manifest."""
    payload = json.dumps(
        [o.to_dict() if hasattr(o, "to_dict") else _as_jsonable(o) for o in objects],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _as_jsonable(obj: Any) -> Any:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def with_sql_model(model: ModelConfig, sql_model: str | None) -> ModelConfig:
    return replace(model, sql_model=sql_model) if sql_model else model


def list_configs(kind: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / kind).glob("*.yaml"))
