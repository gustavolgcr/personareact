"""Tests for the corpus and the joins built on it.

The original pipeline joined questions to answers to scores by list position.
These tests make the replacement -- a join on ``question_id`` -- fail loudly
when it would previously have shifted silently.
"""

from __future__ import annotations

import pytest

from personareact.corpus.schema import NEUTRAL, Answer, Corpus, Question, read_jsonl, write_jsonl
from personareact.paths import DATA_DIR


def make_question(qid: str, neutral: str, profile: str = "Torcedor do Time") -> Question:
    return Question(
        question_id=qid,
        neutral_id=neutral,
        domain="soccer",
        difficulty="easy",
        profile=profile,
        text=f"pergunta {qid}",
    )


class TestCorpus:
    def test_duplicate_ids_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate question_id"):
            Corpus([make_question("a", "a"), make_question("a", "a")])

    def test_unknown_id_raises_with_a_useful_message(self):
        corpus = Corpus([make_question("a", "a")])
        with pytest.raises(KeyError, match="unknown question_id"):
            corpus.get("nope")

    def test_select_filters_and_sorts(self):
        corpus = Corpus([make_question("b", "b"), make_question("a", "a"), make_question("c", "c")])
        assert [q.question_id for q in corpus.select(domain="soccer")] == ["a", "b", "c"]
        assert corpus.select(domain="covid") == []

    def test_neutral_of_resolves(self):
        neutral = make_question("n", "n", profile=NEUTRAL)
        variant = make_question("v", "n")
        corpus = Corpus([neutral, variant])
        assert corpus.neutral_of(variant).question_id == "n"

    def test_roundtrip(self, tmp_path):
        corpus = Corpus([make_question("a", "a"), make_question("b", "a")])
        path = tmp_path / "questions.jsonl"
        assert corpus.save(path) == 2
        assert len(Corpus.load(path)) == 2

    def test_missing_file_explains_how_to_build_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="corpus build"):
            Corpus.load(tmp_path / "absent.jsonl")


class TestRecords:
    def test_answer_roundtrip_keeps_the_failure_flag(self, tmp_path):
        answer = Answer(
            question_id="q",
            run_id="r",
            domain="soccer",
            approach="react",
            model="gpt-4.1-mini",
            profile="Torcedor do Time",
            difficulty="easy",
            question="q?",
            answer="",
            ok=False,
            error="empty answer",
        )
        path = tmp_path / "answers.jsonl"
        write_jsonl(path, [answer])
        restored = Answer.from_dict(next(iter(read_jsonl(path))))
        assert restored.ok is False
        assert restored.error == "empty answer"

    def test_unknown_fields_are_ignored_on_load(self, tmp_path):
        path = tmp_path / "answers.jsonl"
        write_jsonl(
            path,
            [
                {
                    "question_id": "q",
                    "run_id": "r",
                    "domain": "soccer",
                    "approach": "react",
                    "model": "m",
                    "profile": "p",
                    "difficulty": "easy",
                    "question": "?",
                    "answer": "a",
                    "legacy_column": "ignored",
                }
            ],
        )
        assert Answer.from_dict(next(iter(read_jsonl(path)))).answer == "a"


@pytest.mark.skipif(
    not (DATA_DIR / "questions.jsonl").is_file(),
    reason="corpus not built; run scripts/migrate_legacy.py",
)
class TestRealCorpus:
    """Integrity of the migrated thesis corpus."""

    @pytest.fixture(scope="class")
    def corpus(self) -> Corpus:
        return Corpus.load(DATA_DIR / "questions.jsonl")

    def test_size(self, corpus):
        # 2 domains x 2 difficulties x 7 profile sets (6 + neutral) x 10 questions
        assert len(corpus) == 280

    def test_every_variant_resolves_to_its_neutral(self, corpus):
        ids = {q.question_id for q in corpus}
        assert all(q.neutral_id in ids for q in corpus)

    def test_each_neutral_has_exactly_six_variants(self, corpus):
        by_neutral: dict[str, int] = {}
        for question in corpus:
            if question.profile != NEUTRAL:
                by_neutral[question.neutral_id] = by_neutral.get(question.neutral_id, 0) + 1
        assert set(by_neutral.values()) == {6}

    def test_neutral_questions_are_flagged(self, corpus):
        assert sum(1 for q in corpus if q.is_neutral) == 40

    def test_no_empty_question_text(self, corpus):
        assert all(q.text.strip() for q in corpus)

    def test_both_difficulties_present(self, corpus):
        assert {q.difficulty for q in corpus} == {"easy", "hard"}
