"""The experiment runner: sweep, resume, account, record.

This is the piece the original implementation did not have. There, an
experimental condition was selected by editing source -- commenting one of two
``DB_PATH`` lines for the domain, editing a string literal for the model,
commenting five of six list entries for the profile -- so a full Experiment 1
sweep (2 domains x 3 approaches x 6 profiles x 2 difficulties x 7 models = 504
cells) was 504 manual edit-save-run-rename cycles with no resume and no record.

Here a sweep is one command, it is restartable, it stops at a cost ceiling, and
it writes a manifest that says exactly what produced the answers.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .agent import Agent
from .config import (
    ApproachConfig,
    CellConfig,
    DomainConfig,
    ExperimentConfig,
    ModelConfig,
    config_fingerprint,
)
from .corpus.schema import Answer, Corpus, append_jsonl, read_jsonl
from .manifest import Manifest, new_run_id, sha256_file
from .paths import DATA_DIR, RESULTS_DIR, ensure_dir
from .prompts import manifest_hashes
from .tools.text_to_sql import verify_database

CORPUS_PATH = DATA_DIR / "questions.jsonl"


class CostLimitExceeded(RuntimeError):
    pass


#: (input, output) tokens per question, measured on the `smoke` experiment with
#: gpt-4.1-mini rather than guessed. The nested Text-to-SQL agent dominates:
#: `act`, which is nothing but a tool call, is the most expensive approach of
#: the six because the agent lists tables and reads schemas before querying.
#: Re-measure with `personareact run --experiment smoke` if the prompts change.
PER_QUESTION_TOKENS: dict[str, tuple[int, int]] = {
    "standard": (250, 50),
    "cot": (300, 220),
    "act": (24000, 400),
    "react": (7400, 360),
    "react_enriched": (5700, 350),
    "personareact": (5500, 390),
    "_default": (7000, 400),
}


@dataclass
class RunOptions:
    run_id: str | None = None
    resume: str | None = None
    limit_questions: int | None = None
    dry_run: bool = False
    cost_limit_usd: float | None = None
    verbose: bool = False
    strict_db: bool = True
    on_progress: Callable[[str], None] | None = None


def _log(options: RunOptions, message: str) -> None:
    if options.on_progress:
        options.on_progress(message)


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    """(cell slug, question_id) pairs already present, for resume."""
    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        slug = "__".join(
            [
                row.get("domain", ""),
                row.get("approach", ""),
                row.get("model", ""),
                row.get("profile_slug", "") or _slug(row.get("profile", "")),
                row.get("difficulty", ""),
            ]
        )
        keys.add((slug, row.get("question_id", "")))
    return keys


def _slug(text: str) -> str:
    from .config import slugify

    return slugify(text) if text else ""


def estimate_cost(cells: Iterable[CellConfig], questions_per_cell: int) -> dict[str, float]:
    """Pre-flight estimate from measured per-approach token counts.

    Its job is to stop someone starting a US$400 sweep by accident, not to be
    accurate to the cent: the figures in ``PER_QUESTION_TOKENS`` come from one
    model on one question, and the nested SQL agent's usage varies a lot with
    how many schema lookups a question needs.
    """
    total_in = total_out = 0.0
    usd = 0.0
    for cell in cells:
        per_question = PER_QUESTION_TOKENS.get(cell.approach.name, PER_QUESTION_TOKENS["_default"])
        n_in = per_question[0] * questions_per_cell
        n_out = per_question[1] * questions_per_cell
        total_in += n_in
        total_out += n_out
        usd += (
            n_in * cell.model.usd_per_1m_input + n_out * cell.model.usd_per_1m_output
        ) / 1_000_000
    return {"input_tokens": total_in, "output_tokens": total_out, "usd": usd}


def run_cells(
    cells: list[CellConfig],
    *,
    corpus: Corpus,
    options: RunOptions,
    experiment: str | None = None,
) -> Manifest:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not options.dry_run:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    run_id = options.resume or options.run_id or new_run_id(experiment or "run")
    directory = ensure_dir(RESULTS_DIR / run_id)
    answers_path = directory / "answers.jsonl"

    manifest = Manifest(run_id=run_id, kind="generate", experiment=experiment)
    manifest.config_fingerprint = config_fingerprint(*cells[:1]) if cells else None
    manifest.corpus = {
        "path": str(CORPUS_PATH.name),
        "questions": len(corpus),
        "sha256": sha256_file(CORPUS_PATH)[:16] if CORPUS_PATH.is_file() else None,
    }

    # Verify every database once, up front, and record its checksum.
    for domain_name in sorted({c.domain.name for c in cells}):
        domain = DomainConfig.load(domain_name)
        report = verify_database(domain, strict=options.strict_db)
        manifest.database_hashes[domain_name] = report["sha256"] or "unverified"
        manifest.prompt_hashes.update(manifest_hashes(domain.prompt_dir))
        if report["missing_views"]:
            manifest.note(
                f"{domain_name}: missing views {report['missing_views']} (strict_db=False)"
            )

    questions_per_cell = options.limit_questions or 10
    estimate = estimate_cost(cells, questions_per_cell)
    manifest.cells_total = len(cells)
    manifest.note(
        f"pre-flight estimate: {len(cells)} cells, "
        f"~{estimate['input_tokens'] / 1e6:.2f}M in / {estimate['output_tokens'] / 1e6:.2f}M out, "
        f"~US$ {estimate['usd']:.2f}"
    )
    _log(options, manifest.notes[-1])

    if options.dry_run:
        manifest.finished_utc = None
        manifest.note("dry run: nothing was executed")
        manifest.save(directory)
        return manifest

    limit = options.cost_limit_usd
    if limit is None:
        env_limit = os.environ.get("PERSONAREACT_COST_LIMIT_USD")
        limit = float(env_limit) if env_limit else None

    done = _existing_keys(answers_path) if options.resume else set()
    if done:
        _log(options, f"resuming {run_id}: {len(done)} answers already on disk")

    started = time.perf_counter()

    for index, cell in enumerate(cells, start=1):
        questions = corpus.select(
            domain=cell.domain.name,
            difficulty=cell.difficulty,
            profile=cell.profile,
            limit=options.limit_questions,
        )
        if not questions:
            manifest.cells_skipped += 1
            manifest.note(f"no questions for {cell.domain.name}/{cell.difficulty}/{cell.profile}")
            continue

        pending = [q for q in questions if (cell.slug, q.question_id) not in done]
        if not pending:
            manifest.cells_skipped += 1
            continue

        _log(
            options,
            f"[{index}/{len(cells)}] {cell.approach.name} · {cell.domain.name} · "
            f"{cell.model.name} · {cell.profile} · {cell.difficulty} "
            f"({len(pending)} questions)",
        )

        agent = Agent(
            approach=cell.approach,
            domain=cell.domain,
            model=cell.model,
            api_key=api_key,
            verbose=options.verbose,
        )

        batch: list[Answer] = []
        for question in pending:
            if limit is not None and manifest.usd >= limit:
                manifest.note(f"cost ceiling US$ {limit:.2f} reached; stopping early")
                _log(options, manifest.notes[-1])
                append_jsonl(answers_path, batch)
                manifest.elapsed_s = round(time.perf_counter() - started, 2)
                manifest.save(directory)
                raise CostLimitExceeded(manifest.notes[-1])

            before = agent.usage.usd
            result = agent.answer(question, profile=cell.profile)
            spent = agent.usage.usd - before

            batch.append(
                Answer(
                    question_id=question.question_id,
                    run_id=run_id,
                    domain=cell.domain.name,
                    approach=cell.approach.name,
                    model=cell.model.name,
                    profile=cell.profile,
                    difficulty=cell.difficulty,
                    question=agent.question_text(question, cell.profile),
                    answer=result.answer,
                    ok=result.ok,
                    error=result.error,
                    iterations=result.iterations,
                    scratchpad=result.scratchpad,
                    sql_calls=result.sql_calls,
                    input_tokens=agent.usage.input_tokens,
                    output_tokens=agent.usage.output_tokens,
                    usd=round(spent, 6),
                    elapsed_s=result.elapsed_s,
                )
            )
            manifest.answers_total += 1
            if not result.ok:
                manifest.answers_failed += 1
                _log(options, f"    ! {question.question_id}: {result.error}")
            manifest.usd += spent

        manifest.input_tokens += agent.usage.input_tokens
        manifest.output_tokens += agent.usage.output_tokens

        # Written per cell, so an interrupted sweep loses at most one cell.
        append_jsonl(answers_path, batch)
        manifest.cells_done += 1
        manifest.elapsed_s = round(time.perf_counter() - started, 2)
        manifest.save(directory)

    manifest.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest.elapsed_s = round(time.perf_counter() - started, 2)
    manifest.save(directory)
    return manifest


def run_experiment(name: str, options: RunOptions) -> Manifest:
    experiment = ExperimentConfig.load(name)
    corpus = Corpus.load(CORPUS_PATH)
    cells = experiment.cells()
    if options.limit_questions is None and experiment.limit_questions:
        options.limit_questions = experiment.limit_questions
    return run_cells(cells, corpus=corpus, options=options, experiment=experiment.name)


def run_single(
    *,
    domain: str,
    approach: str,
    model: str,
    profiles: list[str] | None,
    difficulties: list[str],
    options: RunOptions,
) -> Manifest:
    domain_cfg = DomainConfig.load(domain)
    approach_cfg = ApproachConfig.load(approach)
    model_cfg = ModelConfig.load(model)
    corpus = Corpus.load(CORPUS_PATH)

    cells = [
        CellConfig(
            domain=domain_cfg,
            approach=approach_cfg,
            model=model_cfg,
            profile=profile,
            difficulty=difficulty,
        )
        for profile in (profiles or domain_cfg.profiles)
        for difficulty in difficulties
    ]
    return run_cells(cells, corpus=corpus, options=options)


def load_answers(run_id: str) -> list[Answer]:
    path = RESULTS_DIR / run_id / "answers.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no answers.jsonl in results/{run_id}")
    return [Answer.from_dict(row) for row in read_jsonl(path)]


def list_runs() -> list[str]:
    if not RESULTS_DIR.is_dir():
        return []
    return sorted(p.name for p in RESULTS_DIR.iterdir() if (p / "manifest.json").is_file())
