"""Tests for the similarity metrics.

These exist because the two defects they guard against are *silent*: the code
runs, produces plausible numbers, and the numbers mean the opposite of what the
surrounding prose says. Both happened in the original implementation.
"""

from __future__ import annotations

import pytest

from personareact.eval.similarity import (
    bleu,
    compare,
    edit_distance_normalized,
    edit_similarity,
    levenshtein,
    rouge_l,
)

NEUTRAL = "A velocidade média de Bruno Pacheco no jogo contra o Ceará foi de 1.36 m/s."
FAN = (
    "Olha, meu amigo, o Bruno Pacheco voou em campo contra o Ceará! "
    "Correu demais, foi um leão, deu o sangue pela camisa. Que jogo!"
)
COACH = (
    "A velocidade média registrada por Bruno Pacheco na partida contra o Ceará "
    "foi de 1.36 m/s, valor relevante para a análise de intensidade."
)


class TestDirectionality:
    def test_edit_similarity_is_high_for_identical_text(self):
        assert edit_similarity(NEUTRAL, NEUTRAL) == 1.0

    def test_edit_distance_is_low_for_identical_text(self):
        assert edit_distance_normalized(NEUTRAL, NEUTRAL) == 0.0

    def test_the_two_are_complements(self):
        assert edit_similarity(FAN, NEUTRAL) + edit_distance_normalized(FAN, NEUTRAL) == 1.0

    def test_a_close_paraphrase_scores_as_more_similar_than_a_rewrite(self):
        """The regression the thesis' interpretation inverted.

        The coach answer keeps the neutral wording; the fan answer rewrites it.
        Any metric labelled "similarity" must rank coach above fan, and any
        metric labelled "distance" must rank fan above coach.
        """
        assert edit_similarity(COACH, NEUTRAL) > edit_similarity(FAN, NEUTRAL)
        assert edit_distance_normalized(FAN, NEUTRAL) > edit_distance_normalized(COACH, NEUTRAL)
        assert rouge_l(COACH, NEUTRAL) > rouge_l(FAN, NEUTRAL)

    def test_edit_similarity_tracks_rouge_not_against_it(self):
        """A true distance would be anti-correlated with ROUGE-L.

        In Table 15 of the thesis the "Distância de Edição" column moves *with*
        ROUGE-L across profiles, which is the observable signature of a
        similarity wearing a distance's name.
        """
        pairs = [(COACH, NEUTRAL), (FAN, NEUTRAL)]
        rouge = [rouge_l(h, r) for h, r in pairs]
        similarity = [edit_similarity(h, r) for h, r in pairs]
        assert (rouge[0] > rouge[1]) == (similarity[0] > similarity[1])


class TestArgumentOrder:
    def test_bleu_is_asymmetric(self):
        """Swapping hypothesis and reference must change the score.

        The original called ``bleu_score(resposta, resposta_ref)`` on a helper
        whose first argument was the *reference*, so the brevity penalty landed
        on the wrong string -- and on the shorter one, systematically.
        """
        short = "A velocidade foi de 1.36 m/s."
        long = NEUTRAL + " " + COACH
        assert bleu(short, long) != pytest.approx(bleu(long, short), abs=1e-6)

    def test_bleu_penalises_a_short_hypothesis(self):
        short = "A velocidade foi de 1.36 m/s."
        assert bleu(short, NEUTRAL) < bleu(NEUTRAL, NEUTRAL)

    def test_rouge_l_is_one_for_identical_text(self):
        assert rouge_l(NEUTRAL, NEUTRAL) == pytest.approx(1.0)

    def test_every_metric_takes_hypothesis_first(self):
        """compare() must not silently reverse the convention."""
        scores = compare(COACH, NEUTRAL)
        assert scores.rouge_l == pytest.approx(rouge_l(COACH, NEUTRAL))
        assert scores.bleu == pytest.approx(bleu(COACH, NEUTRAL))
        assert scores.edit_similarity == pytest.approx(edit_similarity(COACH, NEUTRAL))


class TestLevenshtein:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("", "", 0),
            ("abc", "abc", 0),
            ("abc", "abd", 1),
            ("kitten", "sitting", 3),
            ("", "abc", 3),
        ],
    )
    def test_known_values(self, a, b, expected):
        assert levenshtein(a, b) == expected


class TestEdgeCases:
    def test_empty_strings_do_not_raise(self):
        assert rouge_l("", NEUTRAL) == 0.0
        assert bleu("", NEUTRAL) == 0.0
        assert edit_similarity("", "") == 1.0

    def test_scores_stay_in_range(self):
        for hypothesis in (FAN, COACH, NEUTRAL, ""):
            scores = compare(hypothesis, NEUTRAL)
            for value in (
                scores.rouge_l,
                scores.bleu,
                scores.edit_similarity,
                scores.edit_distance,
            ):
                assert 0.0 <= value <= 1.0
