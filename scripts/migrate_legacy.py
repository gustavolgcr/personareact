#!/usr/bin/env python3
"""One-time migration of the thesis artefacts into the new layout.

Reads the original twelve-repository tree and produces:

  data/questions.jsonl                    the corpus, with stable question ids
  results/legacy-thesis-2025/             the 3,640 generated answers
  results/legacy-thesis-2025-judge/       the G-Eval scores behind chapter 6

Nothing is corrected here. The point of this script is a *faithful* import, so
that the corrected pipeline can be diffed against the thesis numbers rather
than silently replacing them. Defects found on the way in (empty answers, error
sentinels, the profile-blind ReAct baseline) are recorded as flags on the
records, never edited away.

Usage:
    python scripts/migrate_legacy.py --source ~/doutorado/implementacoes
    python scripts/migrate_legacy.py --source ... --check   # report only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personareact.corpus.schema import (  # noqa: E402
    NEUTRAL,
    Answer,
    Question,
    Score,
    write_jsonl,
)

DOMAINS = ("soccer", "covid")
DIFFICULTIES = ("easy", "hard")

#: Legacy model snapshot -> config name in configs/model/.
MODEL_ALIASES = {
    "o3-mini-2025-01-31": "o3-mini",
    "o4-mini-2025-04-16": "o4-mini",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-5-mini-2025-08-07": "gpt-5-mini",
    "gpt-5-nano-2025-08-07": "gpt-5-nano",
}

#: Legacy approach token in filenames -> approach name in configs/approach/.
APPROACH_ALIASES = {
    "personareact": "personareact",
    "react_com_prompt": "react_enriched",
    "react": "react",
}

#: profile slug -> display name, per domain.
PROFILE_NAMES = {
    "soccer": {
        "torcedor_do_time": "Torcedor do Time",
        "tecnico_do_time": "Técnico do Time",
        "narrador_de_futebol": "Narrador de Futebol",
        "influenciador_digital": "Influenciador Digital",
        "fisioterapeuta_do_time": "Fisioterapeuta do Time",
        "jornalista_esportivo": "Jornalista Esportivo",
        "neutral": NEUTRAL,
    },
    "covid": {
        "enfermeiro": "Enfermeiro",
        "familiar_de_um_paciente": "Familiar de um Paciente",
        "gestor_hospitalar": "Gestor Hospitalar",
        "medico": "Médico",
        "paciente": "Paciente",
        "pesquisador_em_saude_publica": "Pesquisador em Saúde Pública",
        "neutral": NEUTRAL,
    },
}

ERROR_SENTINEL = re.compile(r"^\[ERRO[:\]]|^\[Erro ao executar consulta\]")


def slugify(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return stripped.lower().replace(" ", "_")


def question_id(domain: str, difficulty: str, index: int, profile_slug: str) -> str:
    return f"{domain}-{difficulty}-{index:02d}-{profile_slug}"


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


def build_corpus(source: Path, report: list[str]) -> list[Question]:
    """Read the legacy question arrays and attach stable ids.

    The legacy files are bare JSON arrays of strings with no ids and no link
    between a neutral question and its six rewrites -- every join in the
    original pipeline was by list position. Alignment is *asserted* here: if a
    profile file has a different length than its neutral counterpart, the
    migration fails loudly rather than shifting every downstream comparison.
    """
    questions: list[Question] = []
    base = source / "personareact" / "src" / "data" / "questions"

    for domain in DOMAINS:
        for difficulty in DIFFICULTIES:
            neutral_path = base / domain / f"questions_{domain}_{difficulty}_neutral.json"
            if not neutral_path.is_file():
                report.append(f"MISSING neutral corpus: {neutral_path}")
                continue
            neutral_texts = json.loads(neutral_path.read_text(encoding="utf-8"))

            for index, text in enumerate(neutral_texts, start=1):
                questions.append(
                    Question(
                        question_id=question_id(domain, difficulty, index, "neutral"),
                        neutral_id=question_id(domain, difficulty, index, "neutral"),
                        domain=domain,
                        difficulty=difficulty,
                        profile=NEUTRAL,
                        text=text,
                    )
                )

            for slug, display in PROFILE_NAMES[domain].items():
                if slug == "neutral":
                    continue
                path = base / domain / f"questions_{domain}_{difficulty}_{slug}.json"
                if not path.is_file():
                    report.append(f"MISSING profile corpus: {path}")
                    continue
                texts = json.loads(path.read_text(encoding="utf-8"))
                if len(texts) != len(neutral_texts):
                    raise SystemExit(
                        f"ALIGNMENT FAILURE: {path.name} has {len(texts)} questions but "
                        f"the neutral set has {len(neutral_texts)}. The legacy pipeline "
                        f"joined these by position, so this would have silently "
                        f"misaligned every comparison."
                    )
                for index, text in enumerate(texts, start=1):
                    questions.append(
                        Question(
                            question_id=question_id(domain, difficulty, index, slug),
                            neutral_id=question_id(domain, difficulty, index, "neutral"),
                            domain=domain,
                            difficulty=difficulty,
                            profile=display,
                            text=text,
                        )
                    )
    return questions


# --------------------------------------------------------------------------- #
# answers
# --------------------------------------------------------------------------- #

ANSWER_RE = re.compile(
    r"^answers_(?P<approach>personareact|react_com_prompt|react)_"
    r"(?P<domain>soccer|covid)_(?P<difficulty>easy|hard)_"
    r"(?:(?P<profile>.+?)_)?(?P<model>(?:o3|o4|gpt)-.+)\.json$"
)


def build_answers(source: Path, report: list[str]) -> list[Answer]:
    answers: list[Answer] = []
    counters: dict[str, int] = defaultdict(int)
    base = source / "experiments" / "src" / "data" / "answers_experiment_1"

    for domain in DOMAINS:
        directory = base / domain
        if not directory.is_dir():
            report.append(f"MISSING answers dir: {directory}")
            continue

        for path in sorted(directory.glob("*.json")):
            match = ANSWER_RE.match(path.name)
            if not match:
                report.append(f"UNPARSED filename: {path.name}")
                continue

            approach = APPROACH_ALIASES[match["approach"]]
            difficulty = match["difficulty"]
            model_raw = match["model"]
            model = MODEL_ALIASES.get(model_raw)
            if model is None:
                report.append(f"UNKNOWN model in {path.name}: {model_raw}")
                continue

            profile_slug = match["profile"]
            if profile_slug is None:
                # The plain-ReAct condition has no profile in its filename
                # because it was run once per (difficulty, model) against the
                # NEUTRAL questions -- see docs/ERRATA.md, SV-03.
                profile_slug = "neutral"
            profile = PROFILE_NAMES[domain].get(profile_slug)
            if profile is None:
                report.append(f"UNKNOWN profile in {path.name}: {profile_slug}")
                continue

            rows = json.loads(path.read_text(encoding="utf-8"))
            for index, row in enumerate(rows, start=1):
                text = (row.get("answer") or "").strip()
                ok = bool(text) and not ERROR_SENTINEL.match(text)
                error = None
                if not text:
                    error = "empty answer (legacy)"
                    counters["empty"] += 1
                elif not ok:
                    error = f"error sentinel (legacy): {text[:60]}"
                    counters["sentinel"] += 1

                answers.append(
                    Answer(
                        question_id=question_id(domain, difficulty, index, profile_slug),
                        run_id="legacy-thesis-2025",
                        domain=domain,
                        approach=approach,
                        model=model,
                        profile=profile,
                        difficulty=difficulty,
                        question=row.get("question", ""),
                        answer=text,
                        ok=ok,
                        error=error,
                    )
                )
                counters["total"] += 1

    report.append(
        f"answers: {counters['total']} imported, "
        f"{counters['empty']} empty, {counters['sentinel']} error sentinels"
    )
    return answers


# --------------------------------------------------------------------------- #
# judge scores
# --------------------------------------------------------------------------- #

SCORE_COLUMNS = {
    "score_personareact": "personareact",
    "score_react_com_prompt": "react_enriched",
    "score_react": "react",
}
REASON_COLUMNS = {
    "personareact": "reason_personareact",
    "react_enriched": "reason_react_com_prompt",
    "react": "reason_react",
}
CRITERION_ALIASES = {
    "Linguagem e Estilo Apropriados": "language_and_style",
    "Interpretação e Explicação dos Dados": "data_interpretation",
    "Coerência do Estilo ao Longo do Texto": "style_consistency",
}


def build_scores(source: Path, corpus: list[Question], report: list[str]) -> list[Score]:
    """Import the G-Eval results that back chapter 6.

    The legacy result rows key on the question *text*, so they are matched back
    to question ids through the corpus. Rows whose text cannot be matched are
    reported, never dropped silently.
    """
    by_text: dict[tuple[str, str, str], str] = {}
    for q in corpus:
        by_text[(q.domain, q.difficulty, q.text.strip())] = q.question_id

    scores: list[Score] = []
    unmatched = 0
    base = source / "experiments" / "src" / "results" / "experiment_1"

    for domain in DOMAINS:
        path = base / f"g_eval_experiment_1_final_{domain}.json"
        if not path.is_file():
            report.append(f"MISSING judge results: {path}")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))

        for row in rows:
            key = (row["dataset"], row["dificuldade"], row["pergunta"].strip())
            qid = by_text.get(key)
            if qid is None:
                unmatched += 1
                continue
            criterion = CRITERION_ALIASES.get(row["criterio"], row["criterio"])
            profile = PROFILE_NAMES[domain].get(row["perfil"], row["perfil"])

            for column, approach in SCORE_COLUMNS.items():
                values = row.get(column) or []
                reasons = row.get(REASON_COLUMNS[approach]) or []
                for repetition, value in enumerate(values, start=1):
                    scores.append(
                        Score(
                            question_id=qid,
                            run_id="legacy-thesis-2025",
                            judge_run_id="legacy-thesis-2025-judge",
                            domain=row["dataset"],
                            approach=approach,
                            model=MODEL_ALIASES.get(row["modelo"], row["modelo"]),
                            profile=profile,
                            difficulty=row["dificuldade"],
                            criterion=criterion,
                            repetition=repetition,
                            score=float(value),
                            reason=(reasons[repetition - 1] if repetition <= len(reasons) else ""),
                            judge_model="gpt-4o (deepeval default, unpinned)",
                        )
                    )

    report.append(f"scores: {len(scores)} imported, {unmatched} rows unmatched to a question id")
    return scores


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="path to the legacy tree")
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    source: Path = args.source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"source not found: {source}")

    report: list[str] = []
    corpus = build_corpus(source, report)
    answers = build_answers(source, report)
    scores = build_scores(source, corpus, report)

    report.insert(0, f"corpus: {len(corpus)} questions")

    print("\n".join(f"  {line}" for line in report))

    if args.check:
        print("\n--check: nothing written")
        return 0

    corpus_path = REPO_ROOT / "data" / "questions.jsonl"
    write_jsonl(corpus_path, sorted(corpus, key=lambda q: q.question_id))

    answers_dir = REPO_ROOT / "results" / "legacy-thesis-2025"
    write_jsonl(answers_dir / "answers.jsonl", answers)
    _write_legacy_manifest(answers_dir, "generate", len(answers), report)

    scores_dir = REPO_ROOT / "results" / "legacy-thesis-2025-judge"
    write_jsonl(scores_dir / "scores.jsonl", scores)
    _write_legacy_manifest(scores_dir, "judge", len(scores), report)

    print(f"\nwrote {corpus_path.relative_to(REPO_ROOT)}")
    print(f"wrote {(answers_dir / 'answers.jsonl').relative_to(REPO_ROOT)}")
    print(f"wrote {(scores_dir / 'scores.jsonl').relative_to(REPO_ROOT)}")
    return 0


def _write_legacy_manifest(directory: Path, kind: str, count: int, report: list[str]) -> None:
    payload: dict[str, Any] = {
        "run_id": directory.name,
        "kind": kind,
        "provenance": "imported from the original twelve-repository tree by scripts/migrate_legacy.py",
        "faithful": True,
        "records": count,
        "known_defects": [
            "SV-03: the `react` rows answered the NEUTRAL question, not the profile's phrasing",
            "SV-04: the judge saw a different context for `react` than for the other approaches",
            "SV-05: one react answer set was scored once per profile and counted six times",
            "SV-06: the covid runs used a football Text-to-SQL prompt and football profile examples",
            "SV-09: empty answers and `[ERRO: 'thought']` sentinels were scored as answers",
            "SV-13: the judge model was deepeval's unpinned `gpt-4o` default",
            "SV-14: PersonaReAct and the ReAct baselines queried different builds of soccer.sqlite",
        ],
        "migration_report": report,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
