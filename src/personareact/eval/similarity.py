"""Textual and semantic similarity between a personalized answer and its neutral
reference.

**Every function here takes ``(hypothesis, reference)`` in that order.** That
convention is enforced by a test, because getting it wrong is silent and the
original implementation got it wrong for BLEU:

    def bleu_score(a, b):
        return sentence_bleu([a.split()], b.split(), ...)   # a is the REFERENCE

    ...
    "bleu":    bleu_score(resposta, resposta_ref)   # personalized as reference
    "rouge_l": rouge_l_score(resposta, resposta_ref) # personalized as hypothesis

so BLEU and ROUGE used opposite conventions in the same table, and BLEU's
brevity penalty was applied to the *neutral* answer -- systematically the
shorter of the two, and shorter in a way that correlates with the effect being
measured. Section 5.3 of the thesis states the neutral answer is always the
reference.

The second fix concerns naming. The original computed

    textdistance.levenshtein.normalized_similarity(a, b)

and labelled the column "Distância de Edição". That function returns
``1 - lev/max(len)``: **high means more similar**. Sections 6.2 of the thesis
reads it as a distance ("distância de edição alta indica maior divergência"),
which inverts the interpretation. The evidence is in the thesis' own tables --
the column tracks ROUGE-L *positively* across profiles, which a true distance
could not do. Both directions are exposed here under unambiguous names.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# --------------------------------------------------------------------------- #
# ROUGE-L
# --------------------------------------------------------------------------- #


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(hypothesis: str, reference: str, *, beta: float = 1.0) -> float:
    """LCS-based F-measure, exactly the definition in section 5.3.

    Implemented directly rather than through a third-party package so the
    formula in the thesis and the number in the table are the same object:
    precision = LCS/|hypothesis|, recall = LCS/|reference|, combined with the
    beta-weighted harmonic mean.
    """
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    beta2 = beta * beta
    return ((1 + beta2) * precision * recall) / (precision + beta2 * recall)


# --------------------------------------------------------------------------- #
# BLEU
# --------------------------------------------------------------------------- #


def bleu(hypothesis: str, reference: str, *, max_n: int = 4) -> float:
    """Smoothed sentence-level BLEU of ``hypothesis`` against ``reference``.

    Uses add-one smoothing on the higher-order n-gram counts, which is the
    behaviour of NLTK's ``SmoothingFunction().method4`` family for short
    single-reference sentences and keeps the metric from collapsing to zero
    whenever a 4-gram is missing.
    """
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not hyp or not ref:
        return 0.0

    log_precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngram_counts(hyp, n)
        ref_ngrams = _ngram_counts(ref, n)
        if not hyp_ngrams:
            log_precisions.append(math.log(1e-12))
            continue
        overlap = sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        # add-one smoothing for orders above unigram
        if overlap == 0 and n > 1:
            overlap, total = 1, total + 1
        log_precisions.append(math.log(max(overlap, 1e-12) / total))

    brevity = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    return brevity * math.exp(sum(log_precisions) / max_n)


def _ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Edit
# --------------------------------------------------------------------------- #


def levenshtein(a: str, b: str) -> int:
    """Raw edit distance in characters -- the recursion given in section 5.3."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def edit_similarity(hypothesis: str, reference: str) -> float:
    """``1 - lev/max(len)`` in [0, 1]. **High means similar.**

    This is the quantity the original code computed and mislabelled.
    """
    longest = max(len(hypothesis), len(reference))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(hypothesis, reference) / longest


def edit_distance_normalized(hypothesis: str, reference: str) -> float:
    """``lev/max(len)`` in [0, 1]. **High means different.**

    Use this one when the sentence you are writing says "distance".
    """
    return 1.0 - edit_similarity(hypothesis, reference)


# --------------------------------------------------------------------------- #
# Semantic
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=2)
def _sentence_model(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "semantic similarity needs the `embeddings` extra:\n"
            "    uv pip install -e '.[embeddings]'"
        ) from exc
    return SentenceTransformer(name)


def cosine_similarity(
    hypothesis: str, reference: str, *, model_name: str = "all-mpnet-base-v2"
) -> float:
    """Cosine similarity between sentence embeddings.

    ``all-mpnet-base-v2`` is what the code actually used. Section 5.5 of the
    thesis names ``all-MiniLM-L6-v2`` and spends a paragraph justifying it;
    section 5.3 names mpnet and is the one that matches the implementation.
    """
    model = _sentence_model(model_name)
    vectors = model.encode([hypothesis, reference], normalize_embeddings=True)
    return float(vectors[0] @ vectors[1])


def cosine_distance(hypothesis: str, reference: str, **kwargs: object) -> float:
    return 1.0 - cosine_similarity(hypothesis, reference, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #


@dataclass
class SimilarityScores:
    rouge_l: float
    bleu: float
    edit_similarity: float
    edit_distance: float
    cosine_similarity: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def compare(hypothesis: str, reference: str, *, semantic: bool = False) -> SimilarityScores:
    """All textual metrics at once, with the argument order fixed in one place."""
    return SimilarityScores(
        rouge_l=rouge_l(hypothesis, reference),
        bleu=bleu(hypothesis, reference),
        edit_similarity=edit_similarity(hypothesis, reference),
        edit_distance=edit_distance_normalized(hypothesis, reference),
        cosine_similarity=cosine_similarity(hypothesis, reference) if semantic else None,
    )
