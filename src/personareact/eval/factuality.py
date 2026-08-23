"""Does the personalized answer still say what the data says?

Nothing in the original evaluation could answer that. All three G-Eval criteria
measured style, interpretation quality and stylistic consistency; none compared
an answer to the database. That is the strongest objection a reviewer can raise
against a personalization layer -- especially one whose prompt instructs the
model to "evite números crus" for some profiles -- and the artefacts show it
was a real risk: a legacy answer reports "velocidade média ... foi de 1.362
km/h" for a column stored in metres per second, which Quadro 6 of the thesis
renders correctly as 1.36 m/s ~ 4.9 km/h.

Two complementary checks, both cheap and neither needing a judge:

``numeric_grounding``
    Every number the answer states should be traceable to a number the tool
    actually retrieved (allowing for rounding). Ungrounded numbers are
    hallucinated or converted.

``numeric_agreement``
    When a gold value is known for the question, the answer should contain it.

Gold values are optional fields on :class:`~personareact.corpus.schema.Question`
(``gold_sql`` / ``gold_answer``); ``docs/GOLD.md`` describes how to author them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# Matches 1.234, 1,234, 12%, 1.36, 4,9 -- Portuguese and English decimals.
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?![\w])")

# Years, and small integers that are almost always ordinals or counts of items
# in a list rather than retrieved data.
_IGNORED_EXACT = {*(float(y) for y in range(1900, 2101))}


def extract_numbers(text: str) -> list[float]:
    """Numbers stated in the text, normalised to floats."""
    out: list[float] = []
    for raw in _NUMBER.findall(text or ""):
        value = _to_float(raw)
        if value is not None:
            out.append(value)
    return out


def _to_float(raw: str) -> float | None:
    candidate = raw
    if "," in candidate and "." in candidate:
        # 1.234,56 (pt-BR) vs 1,234.56 (en)
        if candidate.rfind(",") > candidate.rfind("."):
            candidate = candidate.replace(".", "").replace(",", ".")
        else:
            candidate = candidate.replace(",", "")
    elif "," in candidate:
        candidate = candidate.replace(",", ".")
    try:
        return float(candidate)
    except ValueError:
        return None


def _matches(value: float, candidates: list[float], *, rel_tol: float = 0.02) -> bool:
    """Rounding-tolerant membership.

    A profile that is told to round ("pouco mais de 12 minutos" rather than
    "732 segundos") must not be penalised, so a stated number counts as
    grounded when it is within 2% of a retrieved one, or when it is the
    retrieved one truncated to fewer decimals.
    """
    for candidate in candidates:
        if candidate == value:
            return True
        if candidate != 0 and abs(candidate - value) / abs(candidate) <= rel_tol:
            return True
        for digits in range(0, 4):
            if round(candidate, digits) == value:
                return True
    return False


@dataclass
class FactualityReport:
    stated: int
    grounded: int
    ungrounded: list[float]
    grounding_rate: float | None
    gold_present: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def numeric_grounding(answer: str, observations: list[str]) -> FactualityReport:
    """Fraction of the answer's numbers that appear in what the tool retrieved."""
    retrieved: list[float] = []
    for observation in observations:
        retrieved.extend(extract_numbers(observation))

    stated = [v for v in extract_numbers(answer) if v not in _IGNORED_EXACT]
    if not stated:
        return FactualityReport(stated=0, grounded=0, ungrounded=[], grounding_rate=None)

    ungrounded = [v for v in stated if not _matches(v, retrieved)]
    grounded = len(stated) - len(ungrounded)
    return FactualityReport(
        stated=len(stated),
        grounded=grounded,
        ungrounded=ungrounded,
        grounding_rate=grounded / len(stated),
    )


def numeric_agreement(answer: str, gold_answer: str | None) -> bool | None:
    """Whether the gold value is present in the answer. ``None`` when no gold."""
    if not gold_answer:
        return None
    gold_values = extract_numbers(gold_answer)
    if not gold_values:
        return gold_answer.strip().lower() in (answer or "").lower()
    stated = extract_numbers(answer)
    return all(_matches(value, stated) for value in gold_values)


def observations_from_scratchpad(scratchpad: list[str]) -> list[str]:
    return [line[len("Observation: ") :] for line in scratchpad if line.startswith("Observation: ")]


def assess(
    answer: str,
    *,
    scratchpad: list[str] | None = None,
    sql_calls: list[dict[str, Any]] | None = None,
    gold_answer: str | None = None,
) -> FactualityReport:
    observations: list[str] = []
    if scratchpad:
        observations.extend(observations_from_scratchpad(scratchpad))
    for call in sql_calls or []:
        output = call.get("output")
        if isinstance(output, str):
            observations.append(output)

    report = numeric_grounding(answer, observations)
    report.gold_present = numeric_agreement(answer, gold_answer)
    return report
