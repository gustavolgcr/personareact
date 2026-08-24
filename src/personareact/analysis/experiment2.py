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

import numpy as np
import pandas as pd

from ..corpus.schema import NEUTRAL, Answer, Corpus
from ..eval.similarity import compare
from ..paths import DATA_DIR, FIGURES_DIR, TABLES_DIR, ensure_dir

METRICS = ["rouge_l", "bleu", "edit_similarity", "edit_distance"]

#: t-SNE needs more points than dimensions to say anything; below this a
#: projection is noise wearing a chart.
MIN_PROJECTION_POINTS = 6


def build_pairs(
    answers: list[Answer],
    corpus: Corpus,
    *,
    reference_approach: str | None = None,
    semantic: bool = False,
) -> tuple[pd.DataFrame, str]:
    """One row per (personalized answer, its neutral reference).

    ``semantic`` adds the cosine similarity between sentence embeddings, which
    section 5.5 of the thesis uses to check that personalization did not
    destroy the meaning. It needs the ``embeddings`` extra and is off by default
    because that pulls PyTorch.
    """
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
                "_hypothesis": answer.answer,
                "_reference": reference.answer,
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
    if semantic and not frame.empty:
        frame["cosine_similarity"] = _batch_cosine(
            frame["_hypothesis"].tolist(), frame["_reference"].tolist()
        )
        frame["cosine_distance"] = 1.0 - frame["cosine_similarity"]
    frame = frame.drop(columns=[c for c in ("_hypothesis", "_reference") if c in frame])

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
    semantic: bool = False,
) -> dict[str, Path]:
    from ..runner import load_answers

    answers = load_answers(run_id)
    corpus = Corpus.load(DATA_DIR / "questions.jsonl")
    frame, note = build_pairs(
        answers, corpus, reference_approach=reference_approach, semantic=semantic
    )
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
    for figure in plot_question_bars(frame, figures_dir):
        written[figure.stem] = figure
    for figure in plot_metric_boxplots(frame, figures_dir):
        written[figure.stem] = figure
    if semantic:
        for figure in plot_projection(frame, answers, figures_dir):
            written[figure.stem] = figure

    return written


# --------------------------------------------------------------------------- #
# Semantic similarity and the projection (thesis Figures 23-26)
# --------------------------------------------------------------------------- #


def _batch_cosine(hypotheses: list[str], references: list[str]) -> list[float]:
    """Cosine similarity for many pairs, encoding once instead of per pair.

    ``all-mpnet-base-v2`` is the model the code actually used. Section 5.5 of
    the thesis names ``all-MiniLM-L6-v2`` and justifies it at length; section
    5.3 names mpnet and is the one that matches the implementation.
    """
    from ..eval.similarity import _sentence_model

    model = _sentence_model("all-mpnet-base-v2")
    unique = list(dict.fromkeys(hypotheses + references))
    vectors = model.encode(
        unique, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    )
    index = {text: vector for text, vector in zip(unique, vectors, strict=True)}
    return [float(index[h] @ index[r]) for h, r in zip(hypotheses, references, strict=True)]


def plot_metric_boxplots(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Thesis Figures 22 and 31: the overall distribution of each metric.

    One panel per metric, one box per model, so the spread is visible rather
    than collapsed into a mean and a standard deviation.
    """
    plt = _pyplot()
    if plt is None:
        return []

    written: list[Path] = []
    labels = {
        "rouge_l": "ROUGE-L",
        "bleu": "BLEU",
        "edit_distance": "Edit distance (1 = fully rewritten)",
        "edit_similarity": "Edit similarity (1 = identical)",
    }
    for domain in sorted(frame["domain"].unique()):
        subset = frame[frame["domain"] == domain]
        models = sorted(subset["model"].unique())
        metrics = [m for m in METRICS if m in subset]
        figure, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.6))
        axes = axes if len(metrics) > 1 else [axes]
        for axis, metric in zip(axes, metrics, strict=True):
            axis.boxplot(
                [subset[subset["model"] == m][metric].dropna() for m in models],
                tick_labels=models,
                showmeans=True,
            )
            axis.set_title(labels.get(metric, metric))
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.3)
            axis.tick_params(axis="x", rotation=25)
        figure.suptitle(f"Distribution of textual divergence from the neutral answer — {domain}")
        figure.tight_layout()
        path = ensure_dir(output_dir) / f"e2_{domain}_distribution_boxplots.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        written.append(path)
    return written


def plot_question_bars(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Thesis Figures 18-19 and 27-28: mean metric per question."""
    plt = _pyplot()
    if plt is None:
        return []

    written: list[Path] = []
    for domain in sorted(frame["domain"].unique()):
        for model in sorted(frame[frame["domain"] == domain]["model"].unique()):
            subset = frame[(frame["domain"] == domain) & (frame["model"] == model)]
            if subset.empty:
                continue
            means = subset.groupby("neutral_id")[["rouge_l", "bleu", "edit_distance"]].mean()
            means.index = [i.split("-")[-2] if "-" in i else i for i in means.index]
            axis = means.plot(kind="bar", figsize=(12, 5), width=0.8, rot=60)
            axis.set_xlabel("Question")
            axis.set_ylabel("Mean")
            axis.set_title(f"Divergence from the neutral answer, per question — {domain} / {model}")
            axis.legend(frameon=False)
            axis.grid(axis="y", alpha=0.3)
            axis.figure.tight_layout()
            path = ensure_dir(output_dir) / f"e2_{domain}_{model}_by_question.png"
            axis.figure.savefig(path, dpi=200)
            plt.close(axis.figure)
            written.append(path)
    return written


def plot_projection(
    frame: pd.DataFrame, answers: list[Answer], output_dir: Path, *, seed: int = 20250820
) -> list[Path]:
    """Thesis Figures 23-26: answers projected to 2D, coloured by profile.

    The thesis highlights the closest and farthest pair inside each cluster.
    Here the pairs are computed in the embedding space, before the projection,
    so they are real neighbours rather than artefacts of t-SNE's layout -- a
    projection is free to place unrelated points next to each other.
    """
    plt = _pyplot()
    if plt is None:
        return []
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return []

    # Decide there is something to project *before* loading a 400 MB sentence
    # model. Checking afterwards was both wasteful and a latent bug: it turned
    # "nothing to plot" into an ImportError whenever the optional `embeddings`
    # extra was absent, which is exactly how CI caught it.
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for domain in sorted(frame["domain"].unique()):
        for model_name in sorted(frame[frame["domain"] == domain]["model"].unique()):
            subset = frame[(frame["domain"] == domain) & (frame["model"] == model_name)]
            if len(subset) >= MIN_PROJECTION_POINTS:
                groups.append((domain, model_name, subset))
    if not groups:
        return []

    from ..eval.similarity import _sentence_model

    model = _sentence_model("all-mpnet-base-v2")
    by_id = {(a.question_id, a.approach, a.model): a for a in answers if a.ok}

    written: list[Path] = []
    for domain, model_name, subset in groups:
        texts, profiles = [], []
        for _, row in subset.iterrows():
            answer = by_id.get((row["question_id"], row["approach"], row["model"]))
            if answer is not None:
                texts.append(answer.answer)
                profiles.append(row["profile"])
        if len(texts) < MIN_PROJECTION_POINTS:
            continue

        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        perplexity = min(10, max(2, (len(texts) - 1) // 3))
        projected = TSNE(
            n_components=2, perplexity=perplexity, random_state=seed, init="pca"
        ).fit_transform(vectors)

        similarity = vectors @ vectors.T
        np.fill_diagonal(similarity, -np.inf)
        closest = np.unravel_index(np.argmax(similarity), similarity.shape)
        np.fill_diagonal(similarity, np.inf)
        farthest = np.unravel_index(np.argmin(similarity), similarity.shape)

        figure, axis = plt.subplots(figsize=(9, 7))
        for profile in sorted(set(profiles)):
            mask = [p == profile for p in profiles]
            axis.scatter(projected[mask, 0], projected[mask, 1], label=profile, s=42, alpha=0.75)
        for pair, colour, style, label in (
            (closest, "tab:blue", "-", "closest pair"),
            (farthest, "tab:red", "--", "farthest pair"),
        ):
            axis.plot(
                projected[list(pair), 0],
                projected[list(pair), 1],
                colour,
                linestyle=style,
                linewidth=2,
                label=label,
            )
        axis.set_title(
            f"Personalized answers in embedding space — {domain} / {model_name}\n"
            "t-SNE projection; pairs measured on the embeddings, not the projection"
        )
        axis.legend(frameon=False, fontsize=8, loc="best")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.tight_layout()
        path = ensure_dir(output_dir) / f"e2_{domain}_{model_name}_projection.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        written.append(path)
    return written


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt
