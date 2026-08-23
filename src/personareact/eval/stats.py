"""Inferential statistics for the judge scores.

The original analysis computed means and standard deviations and stopped there.
``scipy`` was a declared dependency and never imported. The headline claims of
the thesis -- H2, H3 and H4 -- rest on gaps of 0.01 to 0.2 points on a 0-10
scale with no test attached, and an IQR filter deleted out-of-range scores
before aggregation, which removes precisely the disagreements that would have
shown up as variance.

The aggregation was also wrong in shape: it grouped by criterion alone, pooling
7 models x 6 profiles x 2 difficulties x 20 questions x 5 repetitions into one
mean and one "DP", so the reported standard deviation mixes judge repetition
noise with between-question and between-profile variance and answers no
question.

What is done here instead:

* the **question** is the unit of analysis. Judge repetitions are averaged
  within a question first, so five repetitions of the same answer never count
  as five observations;
* comparisons are **paired** on ``question_id`` -- the same question, judged
  under two approaches;
* significance from the Wilcoxon signed-rank test (no normality assumption on
  bounded ordinal scores), with **Holm** correction across the comparison
  family;
* effect size as the **rank-biserial correlation**, plus a bootstrap CI on the
  mean paired difference. A p-value alone says nothing about whether a 0.03
  difference matters.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass
class PairedResult:
    comparison: str
    criterion: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float
    p_holm: float
    effect_size: float
    effect_label: str
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scores_to_frame(scores: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(scores)


def per_question_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse judge repetitions, so one answer contributes one observation."""
    keys = ["domain", "approach", "model", "profile", "difficulty", "criterion", "question_id"]
    return frame.groupby(keys, as_index=False)["score"].mean()


def wilcoxon(a: list[float], b: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test, falling back to a normal approximation.

    Uses ``scipy`` when available; the fallback keeps the analysis runnable
    without the optional ``analysis`` extra installed.
    """
    # All-tied series carry no signal; scipy divides by zero on them.
    if all(x == y for x, y in zip(a, b, strict=True)):
        return 0.0, 1.0
    try:
        from scipy.stats import wilcoxon as _wilcoxon

        statistic, p_value = _wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(statistic), float(p_value)
    except ImportError:
        return _wilcoxon_normal(a, b)


def _wilcoxon_normal(a: list[float], b: list[float]) -> tuple[float, float]:
    diffs = [x - y for x, y in zip(a, b, strict=True) if x != y]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0
    ranked = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    for position, index in enumerate(ranked, start=1):
        ranks[index] = float(position)
    w_plus = sum(r for d, r in zip(diffs, ranks, strict=True) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, ranks, strict=True) if d < 0)
    statistic = min(w_plus, w_minus)
    mean = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd == 0:
        return statistic, 1.0
    z = (statistic - mean) / sd
    p = 2 * (1 - _phi(abs(z)))
    return statistic, min(max(p, 0.0), 1.0)


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def rank_biserial(a: list[float], b: list[float]) -> float:
    """Effect size for a paired rank test, in [-1, 1]."""
    diffs = [x - y for x, y in zip(a, b, strict=True) if x != y]
    if not diffs:
        return 0.0
    positives = sum(1 for d in diffs if d > 0)
    return (2 * positives - len(diffs)) / len(diffs)


def effect_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.1:
        return "negligible"
    if magnitude < 0.3:
        return "small"
    if magnitude < 0.5:
        return "medium"
    return "large"


def bootstrap_ci(
    diffs: list[float], *, iterations: int = 5000, alpha: float = 0.05, seed: int = 20250820
) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iterations):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low = means[int((alpha / 2) * iterations)]
    high = means[min(int((1 - alpha / 2) * iterations), iterations - 1)]
    return (low, high)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        value = (n - rank) * p_values[index]
        running = max(running, min(value, 1.0))
        adjusted[index] = running
    return adjusted


def compare_approaches(
    frame: pd.DataFrame,
    *,
    approach_a: str,
    approach_b: str,
    alpha: float = 0.05,
) -> list[PairedResult]:
    """Paired comparison per criterion, corrected across the criterion family."""
    collapsed = per_question_means(frame)
    results: list[PairedResult] = []
    raw_p: list[float] = []

    for criterion in sorted(collapsed["criterion"].unique()):
        subset = collapsed[collapsed["criterion"] == criterion]
        pivot = subset.pivot_table(
            index=["domain", "model", "profile", "difficulty", "question_id"],
            columns="approach",
            values="score",
        ).dropna(subset=[approach_a, approach_b])
        if pivot.empty:
            continue

        a = pivot[approach_a].tolist()
        b = pivot[approach_b].tolist()
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        statistic, p_value = wilcoxon(a, b)
        low, high = bootstrap_ci(diffs)
        effect = rank_biserial(a, b)

        raw_p.append(p_value)
        results.append(
            PairedResult(
                comparison=f"{approach_a} vs {approach_b}",
                criterion=criterion,
                n_pairs=len(a),
                mean_a=round(sum(a) / len(a), 4),
                mean_b=round(sum(b) / len(b), 4),
                mean_diff=round(sum(diffs) / len(diffs), 4),
                ci_low=round(low, 4),
                ci_high=round(high, 4),
                statistic=round(statistic, 2),
                p_value=p_value,
                p_holm=p_value,  # replaced below
                effect_size=round(effect, 4),
                effect_label=effect_label(effect),
                significant=False,
            )
        )

    for result, adjusted in zip(results, holm(raw_p), strict=True):
        result.p_holm = adjusted
        result.significant = adjusted < alpha

    return results
