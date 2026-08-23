"""The question corpus, and the records produced from it.

The single most consequential structural fix in this rewrite: **every question
has a stable id, and every downstream record carries it.**

The original pipeline stored questions as bare JSON arrays of strings and
joined everything by list position -- ``respostas_1[i]``, ``respostas_3[i]``,
and a dataframe row index in the Experiment 2 notebook. The per-profile
rewrites were produced by ``gpt-4o`` at ``temperature=0.7`` with order
preservation requested only in the prompt text and never verified, so a single
dropped or reordered question would have silently misaligned every comparison
in the thesis with nothing to surface it.

Here a join is a merge on ``question_id`` and a mismatch is an exception.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

NEUTRAL = "neutral"


@dataclass(frozen=True)
class Question:
    """One question, in one phrasing, for one profile.

    ``neutral_id`` links every profile variant back to the neutral question it
    was rewritten from, which is what makes Experiment 2 (personalized vs.
    neutral answer) a well-defined comparison.
    """

    question_id: str
    neutral_id: str
    domain: str
    difficulty: str
    profile: str
    text: str
    gold_sql: str | None = None
    gold_answer: str | None = None

    @property
    def is_neutral(self) -> bool:
        return self.profile == NEUTRAL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Question:
        return cls(
            question_id=raw["question_id"],
            neutral_id=raw["neutral_id"],
            domain=raw["domain"],
            difficulty=raw["difficulty"],
            profile=raw["profile"],
            text=raw["text"],
            gold_sql=raw.get("gold_sql"),
            gold_answer=raw.get("gold_answer"),
        )


@dataclass
class Answer:
    """One generated answer, plus everything needed to audit it.

    The original answer records held four fields (profile, model, question,
    answer). Nothing recorded the prompt version, the parameters, the SQL that
    was actually executed, the iteration count, the token cost, or whether the
    run had failed -- so 18 of the 3,640 answers on disk are empty strings or
    the literal ``[ERRO: 'thought']``, and all of them were sent to the judge
    and scored as if they were answers.
    """

    question_id: str
    run_id: str
    domain: str
    approach: str
    model: str
    profile: str
    difficulty: str
    question: str
    answer: str
    ok: bool = True
    error: str | None = None
    iterations: int = 0
    scratchpad: list[str] = field(default_factory=list)
    sql_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Answer:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class Score:
    """One judge verdict, for one answer, on one criterion, one repetition."""

    question_id: str
    run_id: str
    judge_run_id: str
    domain: str
    approach: str
    model: str
    profile: str
    difficulty: str
    criterion: str
    repetition: int
    score: float
    reason: str = ""
    judge_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Score:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


# --------------------------------------------------------------------------- #
# jsonl io
# --------------------------------------------------------------------------- #


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            payload = record.to_dict() if hasattr(record, "to_dict") else record
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            payload = record.to_dict() if hasattr(record, "to_dict") else record
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


class Corpus:
    """The question set, indexed by id and queryable by axis."""

    def __init__(self, questions: list[Question]):
        self.questions = questions
        self._by_id = {q.question_id: q for q in questions}
        if len(self._by_id) != len(questions):
            seen: set[str] = set()
            dupes = {
                q.question_id for q in questions if q.question_id in seen or seen.add(q.question_id)
            }
            raise ValueError(f"duplicate question_id(s): {sorted(dupes)[:5]}")

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)

    def get(self, question_id: str) -> Question:
        try:
            return self._by_id[question_id]
        except KeyError:
            raise KeyError(f"unknown question_id: {question_id!r}") from None

    def select(
        self,
        *,
        domain: str | None = None,
        difficulty: str | None = None,
        profile: str | None = None,
        limit: int | None = None,
    ) -> list[Question]:
        out = [
            q
            for q in self.questions
            if (domain is None or q.domain == domain)
            and (difficulty is None or q.difficulty == difficulty)
            and (profile is None or q.profile == profile)
        ]
        out.sort(key=lambda q: q.question_id)
        return out[:limit] if limit else out

    def neutral_of(self, question: Question) -> Question:
        return self.get(question.neutral_id)

    @classmethod
    def load(cls, path: Path) -> Corpus:
        if not path.is_file():
            raise FileNotFoundError(
                f"question corpus not found at {path}. Build it with:\n"
                f"    personareact corpus build"
            )
        return cls([Question.from_dict(r) for r in read_jsonl(path)])

    def save(self, path: Path) -> int:
        return write_jsonl(path, sorted(self.questions, key=lambda q: q.question_id))
