"""Tests for the factuality check and the statistics."""

from __future__ import annotations

import pytest

from personareact.eval.factuality import (
    assess,
    extract_numbers,
    numeric_agreement,
    numeric_grounding,
)
from personareact.eval.stats import bootstrap_ci, holm, rank_biserial, wilcoxon


class TestNumberExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("velocidade de 1.36 m/s", [1.36]),
            ("velocidade de 1,36 m/s", [1.36]),
            ("1.234,56 metros", [1234.56]),
            ("sem números aqui", []),
            ("115 segundos e 62 sprints", [115.0, 62.0]),
        ],
    )
    def test_parses_both_decimal_conventions(self, text, expected):
        assert extract_numbers(text) == expected


class TestGrounding:
    OBSERVATION = 'Observation: {"Velocidade_Media": 1.3622634444889454}'

    def test_a_rounded_restatement_is_grounded(self):
        report = numeric_grounding("A velocidade média foi de 1.36 m/s.", [self.OBSERVATION])
        assert report.grounding_rate == 1.0

    def test_a_unit_conversion_is_flagged(self):
        """The defect visible in the thesis artefacts.

        A legacy answer reports "1.362 km/h" for a column stored in metres per
        second. The value is traceable, but the surrounding unit is not -- and
        Quadro 6 of the thesis renders the same figure as 1.36 m/s ~ 4.9 km/h.
        Grounding catches the case where the model invents the converted value.
        """
        report = numeric_grounding("A velocidade média foi de 4.9 km/h.", [self.OBSERVATION])
        assert report.grounding_rate == 0.0
        assert report.ungrounded == [4.9]

    def test_an_invented_number_is_ungrounded(self):
        report = numeric_grounding("Foram 47 sprints no jogo.", [self.OBSERVATION])
        assert 47.0 in report.ungrounded

    def test_a_qualitative_answer_has_no_grounding_rate(self):
        report = numeric_grounding("Ele correu demais, foi um leão em campo!", [self.OBSERVATION])
        assert report.stated == 0
        assert report.grounding_rate is None

    def test_years_are_not_treated_as_data(self):
        report = numeric_grounding("No jogo de 2023 a média foi 1.36.", [self.OBSERVATION])
        assert report.grounding_rate == 1.0

    def test_assess_reads_observations_off_the_scratchpad(self):
        report = assess(
            "A velocidade média foi de 1.36 m/s.",
            scratchpad=["Question: q", self.OBSERVATION, "Standard Answer: ..."],
        )
        assert report.grounding_rate == 1.0


class TestAgreement:
    def test_none_without_a_gold_value(self):
        assert numeric_agreement("qualquer coisa", None) is None

    def test_matches_a_rounded_statement(self):
        assert numeric_agreement("cerca de 1.36 m/s", "1.3622634444889454") is True

    def test_detects_a_wrong_value(self):
        assert numeric_agreement("cerca de 2.5 m/s", "1.36") is False


class TestStats:
    def test_wilcoxon_finds_a_consistent_difference(self):
        a = [9.0, 9.1, 9.2, 8.9, 9.3, 9.0, 9.1, 9.2]
        b = [5.0, 5.1, 4.9, 5.2, 5.0, 4.8, 5.1, 5.0]
        _, p = wilcoxon(a, b)
        assert p < 0.05

    def test_wilcoxon_finds_nothing_in_noise(self):
        a = [9.0, 9.1, 8.9, 9.2, 9.0, 8.8, 9.1, 9.0]
        b = [9.1, 9.0, 9.0, 9.1, 8.9, 9.0, 9.0, 9.1]
        _, p = wilcoxon(a, b)
        assert p > 0.05

    def test_identical_series_are_not_significant(self):
        a = [9.0] * 6
        _, p = wilcoxon(a, list(a))
        assert p == 1.0

    def test_rank_biserial_is_signed(self):
        assert rank_biserial([2, 3, 4], [1, 2, 3]) == 1.0
        assert rank_biserial([1, 2, 3], [2, 3, 4]) == -1.0

    def test_holm_is_monotone_and_conservative(self):
        adjusted = holm([0.01, 0.02, 0.03])
        assert adjusted == sorted(adjusted)
        assert all(adj >= raw for adj, raw in zip(adjusted, [0.01, 0.02, 0.03], strict=True))

    def test_holm_never_exceeds_one(self):
        assert all(value <= 1.0 for value in holm([0.4, 0.5, 0.9]))

    def test_bootstrap_ci_is_deterministic_and_brackets_the_mean(self):
        diffs = [0.1, 0.2, 0.15, 0.05, 0.12, 0.18, 0.09, 0.11]
        first = bootstrap_ci(diffs)
        assert first == bootstrap_ci(diffs)
        mean = sum(diffs) / len(diffs)
        assert first[0] <= mean <= first[1]
