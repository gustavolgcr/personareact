"""Tests for the analysis pipeline, including the thesis reproduction mode."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from personareact.analysis.experiment1 import legacy_iqr_filter, significance_table, summary_tables
from personareact.eval.stats import per_question_means


def make_scores() -> pd.DataFrame:
    """Three criteria per condition, shaped like the real scores.

    The personalized condition sits tightly around 9 on two criteria and has a
    scattered low tail on the third -- which is what really happens, because a
    personalized answer that follows "evite números crus" sometimes drops the
    data entirely and the judge scores it near 3. The baseline is spread from 3
    to 9 on everything.

    That asymmetry is the mechanism: pooling the three criteria makes the
    treatment's low tail look like outliers and deletes it, while the
    baseline's wide bounds keep everything.
    """
    rows = []
    rng = np.random.default_rng(0)
    for question in range(20):
        for repetition in range(1, 6):
            for criterion in ("a", "b", "c"):
                # personalized: tight at 9, except a scattered low tail on "c"
                if criterion == "c" and question % 4 == 0:
                    persona = rng.normal(3.2, 0.6)  # answer dropped the data
                elif criterion == "c":
                    persona = rng.normal(8.4, 0.3)
                else:
                    persona = rng.normal(9.0, 0.1)
                baseline = rng.normal({"a": 6.0, "b": 4.5, "c": 3.5}[criterion], 2.0)
                for approach, score in (("personareact", persona), ("react", baseline)):
                    rows.append(
                        {
                            "question_id": f"q{question:02d}",
                            "domain": "soccer",
                            "approach": approach,
                            "model": "gpt-4.1-mini",
                            "profile": "Torcedor do Time",
                            "difficulty": "easy",
                            "criterion": criterion,
                            "repetition": repetition,
                            "score": float(np.clip(score, 0, 10)),
                        }
                    )
    return pd.DataFrame(rows)


class TestSummaries:
    def test_repetitions_collapse_to_one_observation_per_answer(self):
        """SV-12: five judge repetitions are not five observations."""
        frame = make_scores()
        collapsed = per_question_means(frame)
        assert len(collapsed) == len(frame) / 5

    def test_every_breakdown_keeps_its_grouping_explicit(self):
        tables = summary_tables(make_scores())
        assert set(tables) == {"e1_means", "e1_by_profile", "e1_by_difficulty", "e1_overall"}
        assert {"mean", "sd", "n"} <= set(tables["e1_means"].columns)


class TestLegacyOutlierFilter:
    def test_it_deletes_far_more_of_the_low_variance_condition(self):
        """The asymmetry that inflates the treatments and spares the baseline."""
        frame = make_scores()
        filtered = legacy_iqr_filter(frame)
        kept = filtered.groupby("approach").size()
        original = frame.groupby("approach").size()
        removed = 1 - kept / original
        assert removed["personareact"] > 0.05
        assert removed["react"] < 0.01

    def test_it_raises_the_treatment_on_its_weakest_criterion(self):
        """The mechanism behind PersonaReAct going from 6.90 to 8.81."""
        frame = make_scores()
        before = frame[(frame.approach == "personareact") & (frame.criterion == "c")].score.mean()
        after = legacy_iqr_filter(frame)
        after = after[(after.approach == "personareact") & (after.criterion == "c")].score
        assert after.mean() > before + 0.5

    def test_the_baseline_is_untouched(self):
        frame = make_scores()
        filtered = legacy_iqr_filter(frame)
        for criterion in ("a", "b", "c"):
            raw = frame[(frame.approach == "react") & (frame.criterion == criterion)].score.mean()
            out = filtered[
                (filtered.approach == "react") & (filtered.criterion == criterion)
            ].score.mean()
            assert out == pytest.approx(raw, abs=0.05)

    def test_it_is_not_the_default(self):
        """Reproduction is opt-in; the corrected analysis keeps every score."""
        import inspect

        from personareact.analysis.experiment1 import run

        assert inspect.signature(run).parameters["legacy_outliers"].default is False


class TestSignificance:
    def test_a_real_difference_is_detected_with_a_large_effect(self):
        results = significance_table(make_scores())
        assert not results.empty
        row = results[
            (results.comparison == "personareact vs react") & (results.criterion == "a")
        ].iloc[0]
        assert row.significant
        assert row.effect_label == "large"

    def test_confidence_intervals_bracket_the_difference(self):
        for _, row in significance_table(make_scores()).iterrows():
            assert row.ci_low <= row.mean_diff <= row.ci_high

    def test_holm_correction_is_reported_alongside_the_raw_p(self):
        results = significance_table(make_scores())
        assert (results.p_holm >= results.p_value - 1e-12).all()


class TestExperiment2Figures:
    """The figure types the thesis has and the first port did not."""

    @staticmethod
    def pairs() -> pd.DataFrame:
        rng = np.random.default_rng(1)
        rows = []
        for question in range(6):
            for profile in ("Torcedor do Time", "Técnico do Time", "Fisioterapeuta do Time"):
                rows.append(
                    {
                        "question_id": f"soccer-easy-{question:02d}-p",
                        "neutral_id": f"soccer-easy-{question:02d}-neutral",
                        "domain": "soccer",
                        "approach": "personareact",
                        "model": "gpt-4.1-mini",
                        "profile": profile,
                        "difficulty": "easy",
                        "rouge_l": float(rng.uniform(0.1, 0.3)),
                        "bleu": float(rng.uniform(0, 0.1)),
                        "edit_similarity": float(rng.uniform(0.1, 0.3)),
                        "edit_distance": float(rng.uniform(0.7, 0.9)),
                    }
                )
        return pd.DataFrame(rows)

    def test_per_question_bars_are_produced(self, tmp_path):
        from personareact.analysis.experiment2 import plot_question_bars

        written = plot_question_bars(self.pairs(), tmp_path)
        assert [p.name for p in written] == ["e2_soccer_gpt-4.1-mini_by_question.png"]
        assert written[0].stat().st_size > 0

    def test_distribution_boxplots_are_produced(self, tmp_path):
        from personareact.analysis.experiment2 import plot_metric_boxplots

        written = plot_metric_boxplots(self.pairs(), tmp_path)
        assert [p.name for p in written] == ["e2_soccer_distribution_boxplots.png"]

    def test_the_projection_needs_enough_points(self, tmp_path):
        """Fewer than six answers cannot be projected; it must not raise."""
        from personareact.analysis.experiment2 import plot_projection

        assert plot_projection(self.pairs().head(3), [], tmp_path) == []

    def test_semantic_columns_are_absent_unless_requested(self):
        """The embeddings extra pulls PyTorch, so it stays opt-in."""
        import inspect

        from personareact.analysis.experiment2 import build_pairs, run

        assert inspect.signature(build_pairs).parameters["semantic"].default is False
        assert inspect.signature(run).parameters["semantic"].default is False
