"""``personareact`` -- one command surface for the whole pipeline.

personareact list                       what is configured
personareact data build --domain covid  build a database from source files
personareact data verify                check the databases before a run
personareact corpus stats               inspect the question corpus
personareact run --experiment e1        generate answers (sweep, resumable)
personareact judge --run <id>           score them with G-Eval
personareact analyze --run <id>         regenerate tables and figures
personareact runs                       what is on disk
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import (
    DomainConfig,
    ExperimentConfig,
    JudgeConfig,
    list_configs,
)
from .paths import DATA_DIR, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

console = Console()
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=f"PersonaReAct {__version__} — profile-driven personalization on top of ReAct.",
)
data_app = typer.Typer(no_args_is_help=True, help="Build and verify the domain databases.")
corpus_app = typer.Typer(no_args_is_help=True, help="Inspect the question corpus.")
app.add_typer(data_app, name="data")
app.add_typer(corpus_app, name="corpus")


def _progress(message: str) -> None:
    console.print(message)


# --------------------------------------------------------------------------- #
# list / runs
# --------------------------------------------------------------------------- #


@app.command("list")
def list_command() -> None:
    """Show every configured domain, approach, model, judge and experiment."""
    for kind in ("domain", "approach", "model", "judge", "experiment"):
        table = Table(title=f"configs/{kind}", title_justify="left", show_header=False)
        for name in list_configs(kind):
            table.add_row(name)
        console.print(table)


@app.command("runs")
def runs_command() -> None:
    """List the runs on disk."""
    from .manifest import Manifest
    from .paths import RESULTS_DIR
    from .runner import list_runs

    names = list_runs()
    if not names:
        console.print("[dim]no runs yet[/dim]")
        return

    table = Table(title="results/", title_justify="left")
    for column in ("run", "kind", "cells", "answers", "failed", "USD", "created"):
        table.add_column(column)
    for name in names:
        try:
            manifest = Manifest.load(RESULTS_DIR / name)
        except Exception:
            continue
        table.add_row(
            name,
            manifest.kind,
            f"{manifest.cells_done}/{manifest.cells_total}" if manifest.cells_total else "-",
            str(manifest.answers_total or "-"),
            str(manifest.answers_failed or "-"),
            f"{manifest.usd:.2f}" if manifest.usd else "-",
            (manifest.created_utc or "")[:19],
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


@data_app.command("build")
def data_build(
    domain: Annotated[str, typer.Option(help="soccer | covid")],
    source: Annotated[Path | None, typer.Option(help="directory holding the source files")] = None,
    force: Annotated[bool, typer.Option(help="rebuild even if the database exists")] = False,
) -> None:
    """Build a domain database from its source CSV files."""
    from .datasets import build_domain

    path = build_domain(domain, source=source, force=force, on_progress=_progress)
    console.print(f"[green]built[/green] {path}")
    console.print("Record its checksum with: [bold]personareact data verify --update[/bold]")


@data_app.command("import-legacy")
def data_import_legacy(
    legacy_root: Annotated[Path, typer.Option(help="path to the original implementacoes tree")],
    domain: Annotated[str | None, typer.Option(help="only this domain")] = None,
) -> None:
    """Copy a prebuilt database out of the original tree and repair its views.

    A shortcut for people who already have the databases: it avoids re-reading
    2.5M trajectory points, and it replaces the two broken football views
    (see docs/ERRATA.md, SV-15) without touching a single row of data.
    """
    from .datasets import import_legacy

    for name in [domain] if domain else list_configs("domain"):
        try:
            path = import_legacy(name, legacy_root, on_progress=_progress)
        except FileNotFoundError as exc:
            console.print(f"[yellow]{name}[/yellow]: {exc}")
            continue
        console.print(f"[green]imported[/green] {name} -> {path}")
    console.print("\nNow run: [bold]personareact data verify --update[/bold]")


@data_app.command("verify")
def data_verify(
    domain: Annotated[str | None, typer.Option(help="only this domain")] = None,
    update: Annotated[
        bool, typer.Option(help="write the observed checksum into the config")
    ] = False,
) -> None:
    """Check that each database is present, complete and the expected build."""
    from .tools.text_to_sql import verify_database

    names = [domain] if domain else list_configs("domain")
    for name in names:
        config = DomainConfig.load(name)
        try:
            report = verify_database(config, strict=False)
        except Exception as exc:
            console.print(f"[red]{name}[/red]: {exc}")
            continue

        status = "[green]ok[/green]"
        if report["missing_views"]:
            status = f"[yellow]missing views: {', '.join(report['missing_views'])}[/yellow]"
        elif report["sha256_ok"] is False:
            status = "[red]checksum mismatch[/red]"
        elif report["sha256_ok"] is None:
            status = "[yellow]checksum not pinned[/yellow]"

        console.print(f"{name}: {status}")
        console.print(f"  tables {report['tables']}")
        console.print(f"  views  {report['views'] or '(none)'}")
        console.print(f"  sha256 {report['sha256']}")

        if update and report["sha256"]:
            _pin_checksum(name, report["sha256"])
            console.print(f"  [green]pinned[/green] in configs/domain/{name}.yaml")


def _pin_checksum(domain: str, digest: str) -> None:
    from .paths import CONFIG_DIR

    path = CONFIG_DIR / "domain" / f"{domain}.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("database_sha256:"):
            lines[index] = f"database_sha256: {digest}"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


@corpus_app.command("stats")
def corpus_stats() -> None:
    """Counts and integrity checks for data/questions.jsonl."""
    from .corpus.schema import NEUTRAL, Corpus

    corpus = Corpus.load(DATA_DIR / "questions.jsonl")
    table = Table(title=f"{len(corpus)} questions", title_justify="left")
    for column in ("domain", "difficulty", "profile", "n"):
        table.add_column(column)

    counts: dict[tuple[str, str, str], int] = {}
    for question in corpus:
        key = (question.domain, question.difficulty, question.profile)
        counts[key] = counts.get(key, 0) + 1
    for (domain, difficulty, profile), count in sorted(counts.items()):
        table.add_row(domain, difficulty, profile, str(count))
    console.print(table)

    orphans = [q.question_id for q in corpus if q.neutral_id not in {x.question_id for x in corpus}]
    if orphans:
        console.print(f"[red]{len(orphans)} questions point at a missing neutral_id[/red]")
    else:
        console.print("[green]every question resolves to its neutral counterpart[/green]")

    with_gold = sum(1 for q in corpus if q.gold_answer)
    console.print(
        f"gold answers: {with_gold}/{len(corpus)}"
        + ("" if with_gold else "  [dim](see docs/GOLD.md)[/dim]")
    )
    neutrals = sum(1 for q in corpus if q.profile == NEUTRAL)
    console.print(f"neutral questions: {neutrals}")


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


@app.command("run")
def run_command(
    experiment: Annotated[str | None, typer.Option(help="a configs/experiment/*.yaml")] = None,
    domain: Annotated[str | None, typer.Option()] = None,
    approach: Annotated[str | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option()] = None,
    profile: Annotated[list[str] | None, typer.Option(help="repeatable; default: all")] = None,
    difficulty: Annotated[list[str] | None, typer.Option(help="easy | hard")] = None,
    limit_questions: Annotated[int | None, typer.Option(help="first N questions per cell")] = None,
    resume: Annotated[str | None, typer.Option(help="run id to continue")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="estimate only")] = False,
    cost_limit: Annotated[float | None, typer.Option(help="stop above this many USD")] = None,
    no_strict_db: Annotated[
        bool, typer.Option("--no-strict-db", help="skip checksum enforcement")
    ] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Generate answers, for one cell or a whole experiment."""
    from .runner import CostLimitExceeded, RunOptions, run_experiment, run_single

    options = RunOptions(
        resume=resume,
        limit_questions=limit_questions,
        dry_run=dry_run,
        cost_limit_usd=cost_limit,
        verbose=verbose,
        strict_db=not no_strict_db,
        on_progress=_progress,
    )

    try:
        if experiment:
            manifest = run_experiment(experiment, options)
        else:
            if not (domain and approach and model):
                raise typer.BadParameter("pass --experiment, or all of --domain --approach --model")
            manifest = run_single(
                domain=domain,
                approach=approach,
                model=model,
                profiles=list(profile) if profile else None,
                difficulties=list(difficulty) if difficulty else ["easy", "hard"],
                options=options,
            )
    except CostLimitExceeded as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(code=2) from None

    console.print()
    console.print(f"[bold]run[/bold] {manifest.run_id}")
    console.print(
        f"  cells    {manifest.cells_done}/{manifest.cells_total} "
        f"({manifest.cells_skipped} skipped)"
    )
    console.print(f"  answers  {manifest.answers_total} ({manifest.answers_failed} failed)")
    console.print(f"  tokens   {manifest.input_tokens:,} in / {manifest.output_tokens:,} out")
    console.print(f"  cost     US$ {manifest.usd:.4f}")
    console.print(f"  elapsed  {manifest.elapsed_s:.1f}s")
    if not dry_run:
        console.print(f"\nNext: [bold]personareact judge --run {manifest.run_id}[/bold]")


# --------------------------------------------------------------------------- #
# judge / analyze
# --------------------------------------------------------------------------- #


@app.command("judge")
def judge_command(
    run: Annotated[str, typer.Option(help="generation run id")],
    judge: Annotated[str, typer.Option(help="a configs/judge/*.yaml")] = "geval-gpt-5.6-sol",
    limit: Annotated[int | None, typer.Option(help="first N answers")] = None,
    repetitions: Annotated[
        int | None,
        typer.Option(help="override the judge config's repetition count (cost control)"),
    ] = None,
    resume: Annotated[bool, typer.Option(help="skip scores already on disk")] = False,
    include_failed: Annotated[bool, typer.Option(help="also score ok=False answers")] = False,
) -> None:
    """Score a run's answers with G-Eval."""
    from .eval.judge import JudgeOptions, judge_run

    config = JudgeConfig.load(judge)
    if repetitions is not None:
        if repetitions < 1:
            raise typer.BadParameter("--repetitions must be at least 1")
        config = replace(config, repetitions=repetitions)
        console.print(
            f"[yellow]repetitions overridden to {repetitions} "
            f"(config declares {JudgeConfig.load(judge).repetitions})[/yellow]"
        )
    manifest = judge_run(
        run,
        config,
        JudgeOptions(
            limit=limit,
            resume=resume,
            skip_failed=not include_failed,
            on_progress=_progress,
        ),
    )
    console.print()
    console.print(f"[bold]judge run[/bold] {manifest.run_id}")
    console.print(f"  answers  {manifest.answers_total}")
    console.print(f"  elapsed  {manifest.elapsed_s:.1f}s")
    console.print(f"\nNext: [bold]personareact analyze --run {manifest.run_id}[/bold]")


@app.command("analyze")
def analyze_command(
    run: Annotated[str, typer.Option(help="judge run id (E1) or generation run id (E2)")],
    experiment: Annotated[str, typer.Option(help="e1 | e2")] = "e1",
    reference_approach: Annotated[str | None, typer.Option(help="E2 neutral reference")] = None,
    legacy_outliers: Annotated[
        bool,
        typer.Option(
            "--legacy-outliers",
            help="E1 only: reproduce the thesis IQR filter (see docs/ERRATA.md, SV-12)",
        ),
    ] = False,
    semantic: Annotated[
        bool,
        typer.Option(
            "--semantic",
            help="E2 only: add embedding cosine similarity and the t-SNE projection "
            "(needs the `embeddings` extra)",
        ),
    ] = False,
) -> None:
    """Regenerate every table and figure for a run."""
    if experiment == "e1":
        from .analysis import experiment1

        written = experiment1.run(run, legacy_outliers=legacy_outliers)
        if legacy_outliers:
            console.print(
                "[yellow]legacy outlier filter applied: these reproduce the thesis, "
                "they are not corrected results[/yellow]"
            )
    elif experiment == "e2":
        from .analysis import experiment2

        written = experiment2.run(run, reference_approach=reference_approach, semantic=semantic)
    else:
        raise typer.BadParameter("--experiment must be e1 or e2")

    for name, path in sorted(written.items()):
        console.print(f"  {name:<24} {path.relative_to(PROJECT_ROOT)}")
    console.print(f"\n[green]{len(written)} artefacts written[/green]")


@app.command("describe")
def describe_command(
    experiment: Annotated[str, typer.Argument(help="a configs/experiment/*.yaml")],
) -> None:
    """Show what an experiment would run, without running it."""
    config = ExperimentConfig.load(experiment)
    cells = config.cells()
    console.print(f"[bold]{config.name}[/bold] — {config.description.strip()}")
    console.print(f"\n{len(cells)} cells")
    console.print(f"  domains      {', '.join(config.domains)}")
    console.print(f"  approaches   {', '.join(config.approaches)}")
    console.print(f"  models       {', '.join(config.models)}")
    console.print(f"  difficulties {', '.join(config.difficulties)}")
    console.print(
        f"  profiles     {', '.join(config.profiles) if config.profiles else '(per domain)'}"
    )
    console.print(f"  judge        {config.judge}")


if __name__ == "__main__":
    app()
