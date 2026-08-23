#!/usr/bin/env python3
"""Recompute the Experiment 2 metrics over the thesis answers, corrected.

Same answers, corrected metrics. Isolates exactly what the two Experiment 2
fixes change, without regenerating anything and without a single API call:

* **BLEU** with hypothesis and reference the right way round (SV-02);
* **edit distance** under its real name, next to the similarity the thesis
  actually reported (SV-01).

The published figures are hardcoded below, from Tables 15-18 of the thesis, so
the comparison is checkable at a glance.

    python scripts/compare_with_thesis.py
    python scripts/compare_with_thesis.py --run legacy-thesis-2025 --csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from personareact.analysis.experiment2 import build_pairs  # noqa: E402
from personareact.corpus.schema import Corpus  # noqa: E402
from personareact.paths import DATA_DIR, TABLES_DIR  # noqa: E402
from personareact.runner import load_answers  # noqa: E402

#: Thesis, Tables 17 and 18 -- overall distribution, soccer domain.
THESIS_DISTRIBUTION = {
    "o3-mini": {"rouge_l": 0.1439, "bleu": 0.0088, "edit_reported": 0.1544},
    "gpt-4.1-mini": {"rouge_l": 0.1858, "bleu": 0.0126, "edit_reported": 0.1966},
}

#: Thesis, Table 15 -- per profile, soccer domain, o3 mini.
THESIS_BY_PROFILE_O3 = {
    "Fisioterapeuta do Time": {"rouge_l": 0.110, "bleu": 0.004, "edit_reported": 0.127},
    "Influenciador Digital": {"rouge_l": 0.138, "bleu": 0.006, "edit_reported": 0.157},
    "Jornalista Esportivo": {"rouge_l": 0.166, "bleu": 0.010, "edit_reported": 0.155},
    "Narrador de Futebol": {"rouge_l": 0.104, "bleu": 0.003, "edit_reported": 0.141},
    "Técnico do Time": {"rouge_l": 0.178, "bleu": 0.019, "edit_reported": 0.178},
    "Torcedor do Time": {"rouge_l": 0.167, "bleu": 0.010, "edit_reported": 0.168},
}


def distribution_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, published in THESIS_DISTRIBUTION.items():
        subset = pairs[(pairs["domain"] == "soccer") & (pairs["model"] == model)]
        if subset.empty:
            continue
        rows.append(
            {
                "model": model,
                "n": len(subset),
                "rouge_l_corrected": round(subset["rouge_l"].mean(), 4),
                "rouge_l_thesis": published["rouge_l"],
                "bleu_corrected": round(subset["bleu"].mean(), 4),
                "bleu_thesis": published["bleu"],
                "edit_similarity_corrected": round(subset["edit_similarity"].mean(), 4),
                "edit_reported_as_distance_in_thesis": published["edit_reported"],
                "edit_distance_corrected": round(subset["edit_distance"].mean(), 4),
            }
        )
    return pd.DataFrame(rows)


def profile_table(pairs: pd.DataFrame) -> pd.DataFrame:
    subset = pairs[(pairs["domain"] == "soccer") & (pairs["model"] == "o3-mini")]
    rows = []
    for profile, published in THESIS_BY_PROFILE_O3.items():
        group = subset[subset["profile"] == profile]
        if group.empty:
            continue
        rows.append(
            {
                "profile": profile,
                "n": len(group),
                "rouge_l_corrected": round(group["rouge_l"].mean(), 4),
                "rouge_l_thesis": published["rouge_l"],
                "bleu_corrected": round(group["bleu"].mean(), 4),
                "bleu_thesis": published["bleu"],
                "edit_similarity_corrected": round(group["edit_similarity"].mean(), 4),
                "edit_reported_as_distance_in_thesis": published["edit_reported"],
                "edit_distance_corrected": round(group["edit_distance"].mean(), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="legacy-thesis-2025")
    parser.add_argument("--csv", action="store_true", help="also write the tables to tables/")
    args = parser.parse_args()

    answers = load_answers(args.run)
    corpus = Corpus.load(DATA_DIR / "questions.jsonl")
    pairs, note = build_pairs(answers, corpus)

    distribution = distribution_table(pairs)
    profiles = profile_table(pairs)

    print(f"run: {args.run}")
    print(f"pairing: {note}\n")
    print("OVERALL DISTRIBUTION — soccer (thesis Tables 17 and 18)")
    print(distribution.to_string(index=False))
    print()
    print("PER PROFILE — soccer, o3-mini (thesis Table 15)")
    print(profiles.to_string(index=False))
    print()
    print("Reading the result:")
    print("  ROUGE-L reproduces the published values within tokenisation noise,")
    print("  which is what confirms the migration is faithful.")
    print("  BLEU rises several-fold: the published values were computed with the")
    print("  personalized answer as the *reference*, so the brevity penalty landed")
    print("  on the neutral answer -- systematically the shorter of the two.")
    print("  The column the thesis prints as 'Distância de Edição' matches")
    print("  edit_similarity, not edit_distance. The true distance is ~0.8, not ~0.15,")
    print("  so sentences reading a high value as 'maior divergência' are inverted.")

    if args.csv:
        target = TABLES_DIR / args.run
        target.mkdir(parents=True, exist_ok=True)
        distribution.to_csv(target / "e2_vs_thesis_distribution.csv", index=False)
        profiles.to_csv(target / "e2_vs_thesis_by_profile.csv", index=False)
        print(f"\nwrote {target.relative_to(REPO_ROOT)}/e2_vs_thesis_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
