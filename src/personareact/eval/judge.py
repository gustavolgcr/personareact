"""The LLM-as-a-judge harness.

The central fix here is symmetry. In the original evaluator every condition was
graded with a different context:

    caso_1 = LLMTestCase(..., context=["Perfil do usuário: " + profile])   # personareact
    caso_2 = LLMTestCase(..., context=["Perfil do usuário: " + profile])   # react + prompt
    caso_3 = LLMTestCase(..., context=["Sem perfil de usuário definido."]) # react

and all three criteria listed ``LLMTestCaseParams.CONTEXT`` in their
``evaluation_params``, so the judge read it. When the criterion is "does the
style suit this profile" and the judge is simultaneously told there is no
profile, part of the measured gap belongs to the harness rather than to the
answers.

Here the judging question is identical for every condition: *given this
question and this profile, how good is this answer?* How the answer was
produced never enters the prompt -- which also blinds the judge to the
condition.

Two further changes: the judge model is pinned to a dated snapshot, and answers
flagged ``ok=False`` are never scored (the original graded 15 empty strings and
two ``[ERRO: 'thought']`` sentinels as if they were answers).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..config import JudgeConfig
from ..corpus.schema import Answer, Score, append_jsonl, read_jsonl
from ..manifest import Manifest, new_run_id
from ..paths import RESULTS_DIR, ensure_dir
from .criteria import build_geval_metrics, get_criterion


@dataclass
class JudgeOptions:
    judge_run_id: str | None = None
    resume: bool = False
    limit: int | None = None
    on_progress: Callable[[str], None] | None = None
    skip_failed: bool = True


def build_test_case(answer: Answer):
    """The judge's view of one answer -- identical in shape for every approach."""
    from deepeval.test_case import LLMTestCase

    profile = answer.profile
    return LLMTestCase(
        input=f"Pergunta: {answer.question}\nPerfil do usuário: {profile}",
        actual_output=answer.answer,
        context=[f"Perfil do usuário: {profile}"],
    )


def _existing(path):
    seen: set[tuple[str, str, str, str, str, str, int]] = set()
    for row in read_jsonl(path):
        seen.add(
            (
                row.get("question_id", ""),
                row.get("approach", ""),
                row.get("model", ""),
                row.get("profile", ""),
                row.get("difficulty", ""),
                row.get("criterion", ""),
                int(row.get("repetition", 0)),
            )
        )
    return seen


def judge_run(
    run_id: str,
    config: JudgeConfig,
    options: JudgeOptions | None = None,
) -> Manifest:
    """Score every answer of a generation run."""
    options = options or JudgeOptions()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    from ..runner import load_answers

    answers = load_answers(run_id)
    if options.skip_failed:
        skipped = [a for a in answers if not a.ok]
        answers = [a for a in answers if a.ok]
    else:
        skipped = []
    if options.limit:
        answers = answers[: options.limit]

    judge_run_id = options.judge_run_id or new_run_id(f"judge-{run_id.split('-')[-1]}")
    directory = ensure_dir(RESULTS_DIR / judge_run_id)
    scores_path = directory / "scores.jsonl"

    manifest = Manifest(run_id=judge_run_id, kind="judge")
    manifest.config = {
        "judge": config.name,
        "judge_model": config.model,
        "repetitions": config.repetitions,
        "criteria": config.criteria,
        "source_run": run_id,
        "skipped_failed_answers": len(skipped),
    }
    manifest.note(
        f"judging {len(answers)} answers x {len(config.criteria)} criteria "
        f"x {config.repetitions} repetitions = "
        f"{len(answers) * len(config.criteria) * config.repetitions} calls"
    )
    if skipped:
        manifest.note(
            f"{len(skipped)} answers excluded because ok=False "
            f"(empty or error sentinel); they are recorded in the source run"
        )
    if options.on_progress:
        for note in manifest.notes:
            options.on_progress(note)

    metrics = build_geval_metrics(config.criteria, model=config.model)
    criterion_keys = {get_criterion(k).display_name: k for k in config.criteria}

    done = _existing(scores_path) if options.resume else set()
    started = time.perf_counter()
    batch: list[Score] = []

    for index, answer in enumerate(answers, start=1):
        test_case = build_test_case(answer)
        for metric in metrics:
            key = criterion_keys[metric.name]
            for repetition in range(1, config.repetitions + 1):
                fingerprint = (
                    answer.question_id,
                    answer.approach,
                    answer.model,
                    answer.profile,
                    answer.difficulty,
                    key,
                    repetition,
                )
                if fingerprint in done:
                    continue
                metric.measure(test_case)
                batch.append(
                    Score(
                        question_id=answer.question_id,
                        run_id=run_id,
                        judge_run_id=judge_run_id,
                        domain=answer.domain,
                        approach=answer.approach,
                        model=answer.model,
                        profile=answer.profile,
                        difficulty=answer.difficulty,
                        criterion=key,
                        repetition=repetition,
                        score=round(float(metric.score) * 10, 2),
                        reason=metric.reason or "",
                        judge_model=config.model,
                    )
                )

        if len(batch) >= 50 or index == len(answers):
            append_jsonl(scores_path, batch)
            batch.clear()
            manifest.elapsed_s = round(time.perf_counter() - started, 2)
            manifest.save(directory)
            if options.on_progress:
                options.on_progress(f"  judged {index}/{len(answers)} answers")

    if batch:
        append_jsonl(scores_path, batch)

    manifest.answers_total = len(answers)
    manifest.elapsed_s = round(time.perf_counter() - started, 2)
    manifest.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest.save(directory)
    return manifest
