"""Experiment 1: tables and figures, as a script.

Replaces four Jupyter notebooks that hardcoded ``/Users/<name>/doutorado/...``
in every path and carried ``DATASET = "soccer"`` / ``model_key = "..."`` as
edit-me constants with the alternatives in trailing comments. Nobody but the
author could execute a cell, so no figure in chapter 6 was regenerable by a
reader.

Outputs, all derived from a judge run and written under ``tables/`` and
``figures/``:

* ``e1_means.csv``        mean and SD per (domain, model, approach, criterion)
* ``e1_by_profile.csv``   the same, broken down by profile
* ``e1_by_difficulty.csv``the same, broken down by difficulty
* ``e1_tests.csv``        paired Wilcoxon tests with Holm correction and effect sizes
* ``e1_<domain>_<criterion>.png`` grouped bars per model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..corpus.schema import read_jsonl
from ..eval.criteria import CRITERIA
from ..eval.stats import compare_approaches, per_question_means
from ..paths import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, ensure_dir

PRIMARY_COMPARISONS = [
    ("personareact", "react"),
    ("personareact", "react_enriched"),
    ("react_enriched", "react"),
]


def load_scores(judge_run_id: str) -> pd.DataFrame:
    path = RESULTS_DIR / judge_run_id / "scores.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no scores.jsonl in results/{judge_run_id}")
    frame = pd.DataFrame(list(read_jsonl(path)))
    if frame.empty:
        raise ValueError(f"results/{judge_run_id}/scores.jsonl is empty")
    return frame


def legacy_iqr_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the outlier removal the thesis notebooks applied.

    The notebook laid the scores out with one column per (approach, model) and
    one row per (question, criterion, repetition), then computed IQR bounds
    **per column, with all three criteria pooled**, and replaced anything
    outside them with NaN before grouping by criterion.

    That pooling is what makes the filter asymmetric. The personalized
    conditions score tightly around 9 on style and language but around 7 on
    data interpretation, so pooling makes a whole criterion's worth of
    legitimate scores look like a low tail and deletes it. The ReAct baseline
    is spread from 3 to 9, so its bounds are wide and nothing is removed:

        personareact / o3-mini / covid    19.1% of scores deleted
        react_enriched / o3-mini / covid  18.3% deleted
        react / o3-mini / covid            0.0% deleted

    On "Interpretação e Explicação dos Dados" that lifts PersonaReAct from 6.90
    to 8.81 while leaving ReAct at 3.49 -- so nearly two points of the reported
    gap on that criterion are an artefact of the filter.

    Reproduced here, exactly, so the published tables stay regenerable. It is
    not the default; see ``docs/ERRATA.md`` (SV-12).
    """
    wide = frame.pivot_table(
        index=["question_id", "criterion", "repetition"],
        columns=["domain", "approach", "model"],
        values="score",
    )
    for column in wide.columns:
        q1, q3 = wide[column].quantile(0.25), wide[column].quantile(0.75)
        iqr = q3 - q1
        outside = (wide[column] < q1 - 1.5 * iqr) | (wide[column] > q3 + 1.5 * iqr)
        wide.loc[outside, column] = np.nan

    long = (
        wide.stack(["domain", "approach", "model"], future_stack=True)
        .rename("score")
        .reset_index()
        .dropna(subset=["score"])
    )
    keep = [c for c in ("profile", "difficulty") if c in frame.columns]
    if keep:
        long = long.merge(
            frame[["question_id", *keep]].drop_duplicates("question_id"),
            on="question_id",
            how="left",
        )
    return long


def summary_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregations that keep their variance components separate.

    The original pooled 7 models x 6 profiles x 2 difficulties x 20 questions
    x 5 repetitions into a single mean and a single "DP" grouped by criterion
    alone, so the reported standard deviation mixed judge noise with
    between-question and between-profile variance.

    Here repetitions are collapsed first (one answer, one observation) and each
    breakdown keeps its grouping explicit.
    """
    collapsed = per_question_means(frame)

    def agg(keys: list[str]) -> pd.DataFrame:
        out = (
            collapsed.groupby(keys, as_index=False)["score"]
            .agg(mean="mean", sd="std", n="count")
            .round(4)
        )
        return out.sort_values(keys).reset_index(drop=True)

    return {
        "e1_means": agg(["domain", "model", "approach", "criterion"]),
        "e1_by_profile": agg(["domain", "profile", "approach", "criterion"]),
        "e1_by_difficulty": agg(["domain", "difficulty", "approach", "criterion"]),
        "e1_overall": agg(["approach", "criterion"]),
    }


def significance_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired tests per domain and model, corrected within each family."""
    rows = []
    approaches = set(frame["approach"].unique())
    for domain in sorted(frame["domain"].unique()):
        for model in sorted(frame[frame["domain"] == domain]["model"].unique()):
            subset = frame[(frame["domain"] == domain) & (frame["model"] == model)]
            for a, b in PRIMARY_COMPARISONS:
                if a not in approaches or b not in approaches:
                    continue
                for result in compare_approaches(subset, approach_a=a, approach_b=b):
                    row = result.to_dict()
                    row["domain"] = domain
                    row["model"] = model
                    rows.append(row)
    if not rows:
        return pd.DataFrame()
    columns = [
        "domain",
        "model",
        "comparison",
        "criterion",
        "n_pairs",
        "mean_a",
        "mean_b",
        "mean_diff",
        "ci_low",
        "ci_high",
        "p_value",
        "p_holm",
        "effect_size",
        "effect_label",
        "significant",
    ]
    return pd.DataFrame(rows)[columns]


def plot_criterion_bars(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    collapsed = per_question_means(frame)
    written: list[Path] = []
    display = {k: v["display_name"] for k, v in CRITERIA.items()}

    for domain in sorted(collapsed["domain"].unique()):
        for criterion in sorted(collapsed["criterion"].unique()):
            subset = collapsed[
                (collapsed["domain"] == domain) & (collapsed["criterion"] == criterion)
            ]
            pivot = subset.pivot_table(
                index="model", columns="approach", values="score", aggfunc="mean"
            )
            if pivot.empty:
                continue
            errors = subset.pivot_table(
                index="model", columns="approach", values="score", aggfunc="std"
            ).reindex_like(pivot)

            ax = pivot.plot(kind="bar", figsize=(10, 5), yerr=errors, capsize=3, width=0.78, rot=20)
            ax.set_ylabel("Mean score (0-10)")
            ax.set_xlabel("")
            ax.set_ylim(0, 10)
            ax.set_title(f"{display.get(criterion, criterion)} — {domain}")
            ax.legend(title="Approach", frameon=False)
            ax.grid(axis="y", alpha=0.3)
            ax.figure.tight_layout()

            path = ensure_dir(output_dir) / f"e1_{domain}_{criterion}.png"
            ax.figure.savefig(path, dpi=200)
            ax.figure.clf()
            plt.close("all")
            written.append(path)
    return written


def run(
    judge_run_id: str,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    legacy_outliers: bool = False,
) -> dict[str, Path]:
    frame = load_scores(judge_run_id)
    if legacy_outliers:
        frame = legacy_iqr_filter(frame)
    tables_dir = ensure_dir(tables_dir or TABLES_DIR / judge_run_id)
    figures_dir = ensure_dir(figures_dir or FIGURES_DIR / judge_run_id)

    written: dict[str, Path] = {}
    for name, table in summary_tables(frame).items():
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path

    if legacy_outliers:
        (tables_dir / "e1_NOTE_legacy_outliers.txt").write_text(
            "These tables reproduce the thesis notebooks, including the IQR "
            "outlier filter that deletes ~19% of the personalized conditions' "
            "scores and 0% of the ReAct baseline's. See docs/ERRATA.md, SV-12. "
            "Do not use them as corrected results.\n",
            encoding="utf-8",
        )
        written["e1_NOTE_legacy_outliers"] = tables_dir / "e1_NOTE_legacy_outliers.txt"

    tests = significance_table(frame)
    if not tests.empty:
        path = tables_dir / "e1_tests.csv"
        tests.to_csv(path, index=False)
        written["e1_tests"] = path

    for figure in plot_criterion_bars(frame, figures_dir):
        written[figure.stem] = figure

    return written
