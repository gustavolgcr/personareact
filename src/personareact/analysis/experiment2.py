"""Experiment 2: how far each personalized answer moves from its neutral one.

Two structural fixes over the original notebooks.

**The join.** The original built a dataframe by reading the ReAct answer file
and the six PersonaReAct files and lining them up by row index, raising only if
the *lengths* differed. Here every personalized answer is matched to its
neutral counterpart through ``question_id`` -> ``neutral_id``, so a
misalignment is impossible rather than merely unlikely.

**The reference.** Section 5.5 describes the neutral answer as "gerada pelo
PersonaReAct com o módulo Tailor desativado", but no such switch existed and
the notebooks used the plain-ReAct answer files instead. The Tailor step is now
genuinely skippable (that is section 4.4's backward-compatibility claim), so
running ``personareact`` with ``--profile neutral`` produces exactly the
reference the thesis describes. When a run has no neutral PersonaReAct answers,
this module falls back to the ``react`` answers and says so.

Metrics come from :mod:`personareact.eval.similarity`, where the argument order
is fixed once and tested.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..corpus.schema import NEUTRAL, Answer, Corpus
from ..eval.similarity import compare
from ..paths import DATA_DIR, FIGURES_DIR, TABLES_DIR, ensure_dir

METRICS = ["rouge_l", "bleu", "edit_similarity", "edit_distance"]


def build_pairs(
    answers: list[Answer],
    corpus: Corpus,
    *,
    reference_approach: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """One row per (personalized answer, its neutral reference)."""
    ok = [a for a in answers if a.ok and a.answer.strip()]

    neutral_index: dict[tuple[str, str, str], Answer] = {}
    for answer in ok:
        if answer.profile == NEUTRAL:
            neutral_index[(answer.approach, answer.model, answer.question_id)] = answer

    if reference_approach is None:
        reference_approach = (
            "personareact"
            if any(a.approach == "personareact" for a in neutral_index.values())
            else "react"
        )

    rows = []
    missing_reference = 0
    for answer in ok:
        if answer.profile == NEUTRAL:
            continue
        question = corpus.get(answer.question_id)
        reference = neutral_index.get((reference_approach, answer.model, question.neutral_id))
        if reference is None:
            missing_reference += 1
            continue

        scores = compare(answer.answer, reference.answer)
        rows.append(
            {
                "question_id": answer.question_id,
                "neutral_id": question.neutral_id,
                "domain": answer.domain,
                "approach": answer.approach,
                "model": answer.model,
                "profile": answer.profile,
                "difficulty": answer.difficulty,
                "reference_approach": reference_approach,
                **scores.to_dict(),
            }
        )

    frame = pd.DataFrame(rows)
    note = f"reference = {reference_approach} @ profile={NEUTRAL}"
    if missing_reference:
        note += f"; {missing_reference} answers had no neutral counterpart and were dropped"
    return frame, note


def summary_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def agg(keys: list[str]) -> pd.DataFrame:
        out = frame.groupby(keys)[METRICS].agg(["mean", "std"]).round(4)
        out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
        return out.reset_index()

    overall = frame[METRICS].describe().T.round(4).reset_index()
    overall = overall.rename(columns={"index": "metric"})

    return {
        "e2_by_question": agg(["domain", "model", "neutral_id"]),
        "e2_by_profile": agg(["domain", "model", "profile"]),
        "e2_by_difficulty": agg(["domain", "model", "difficulty"]),
        "e2_distribution": overall,
    }


def plot_profile_bars(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: list[Path] = []
    for domain in sorted(frame["domain"].unique()):
        for model in sorted(frame[frame["domain"] == domain]["model"].unique()):
            subset = frame[(frame["domain"] == domain) & (frame["model"] == model)]
            if subset.empty:
                continue
            means = subset.groupby("profile")[["rouge_l", "bleu", "edit_distance"]].mean()
            ax = means.plot(kind="bar", figsize=(10, 5), width=0.78, rot=25)
            ax.set_ylabel("Mean")
            ax.set_xlabel("")
            ax.set_title(
                f"Divergence from the neutral answer — {domain} / {model}\n"
                "higher ROUGE-L/BLEU = closer;  higher edit distance = further"
            )
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=0.3)
            ax.figure.tight_layout()
            path = ensure_dir(output_dir) / f"e2_{domain}_{model}_by_profile.png"
            ax.figure.savefig(path, dpi=200)
            ax.figure.clf()
            plt.close("all")
            written.append(path)
    return written


def run(
    run_id: str,
    *,
    reference_approach: str | None = None,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Path]:
    from ..runner import load_answers

    answers = load_answers(run_id)
    corpus = Corpus.load(DATA_DIR / "questions.jsonl")
    frame, note = build_pairs(answers, corpus, reference_approach=reference_approach)
    if frame.empty:
        raise ValueError(
            f"no (personalized, neutral) pairs in results/{run_id}. "
            f"Experiment 2 needs a run that includes the `neutral` profile."
        )

    tables_dir = ensure_dir(tables_dir or TABLES_DIR / run_id)
    figures_dir = ensure_dir(figures_dir or FIGURES_DIR / run_id)

    written: dict[str, Path] = {}
    frame.to_csv(tables_dir / "e2_pairs.csv", index=False)
    written["e2_pairs"] = tables_dir / "e2_pairs.csv"

    for name, table in summary_tables(frame).items():
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path

    (tables_dir / "e2_note.txt").write_text(note + "\n", encoding="utf-8")
    written["e2_note"] = tables_dir / "e2_note.txt"

    for figure in plot_profile_bars(frame, figures_dir):
        written[figure.stem] = figure

    return written
