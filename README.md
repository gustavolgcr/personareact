# PersonaReAct

**Profile-driven personalization of language-model answers, on top of the ReAct paradigm.**

PersonaReAct extends [ReAct](https://arxiv.org/abs/2210.03629) with an explicit
final personalization step — a *Tailor* function `τ(p, O*) → M` that adapts the
answer to the functional profile of the person who asked, without touching the
reasoning and acting cycle that produced it. The same question, asked by a team
physiotherapist and by a supporter, yields the same facts in two different
answers.

This repository is the reference implementation and the full experimental
harness for the PhD thesis *"PersonaReAct: Personalização de Respostas em
Modelos de Linguagem com base no Paradigma ReAct"* (UFC, 2025), covering two
case studies: professional football tracking data and COVID-19 hospital
records.

> **Status.** This is a corrected rewrite of the implementation used in the
> thesis. Several defects in the original affect how the published results
> should be read. They are all documented, with evidence, in
> [`docs/ERRATA.md`](docs/ERRATA.md) — read it before citing any number from
> chapter 6. [`docs/REVIEW.md`](docs/REVIEW.md) assesses the original
> LangChain/LangGraph implementation against what chapters 4 and 5 describe.

---

## Quickstart

```bash
git clone https://github.com/gustavolgcr/personareact
cd personareact

# 1. environment
uv venv --python 3.12
uv pip install -e ".[analysis,judge]"

# 2. credentials
cp .env.example .env       # then put your OpenAI key in it

# 3. data (see docs/DATA.md — the COVID database is public, soccer is not)
personareact data build --domain covid

# 4. prove the pipeline works, for a few cents
personareact run --experiment smoke

# 5. see what a full sweep would cost before starting one
personareact run --experiment e1 --dry-run
```

## Running experiments

Every experimental condition is a config file. Nothing is selected by editing
source.

```bash
# one cell: one approach, one model, one profile
personareact run --domain soccer --approach personareact \
                 --model gpt-4.1-mini --profile "Torcedor do Time" \
                 --difficulty easy

# a whole experiment (504 cells for e1)
personareact run --experiment e1

# pick up where an interrupted sweep stopped
personareact run --experiment e1 --resume 2026-08-23T1400Z-e1

# judge the answers of a run
personareact judge --run 2026-08-23T1400Z-e1

# regenerate every table and figure for that run
personareact analyze --run 2026-08-23T1400Z-e1
```

`personareact list` shows every available domain, approach, model, judge and
experiment. `personareact runs` lists the runs on disk.

### The six approaches

| Approach | Reasoning | Tools | Tailor | Profile reaches the model via |
|---|---|---|---|---|
| `standard` | – | – | – | nothing |
| `cot` | yes | – | – | nothing |
| `act` | – | yes | – | nothing |
| `react` | yes | yes | – | nothing |
| `react_enriched` | yes | yes | – | a suffix appended to the question |
| `personareact` | yes | yes | **yes** | the Tailor step |

`standard`, `cot` and `act` correspond to the paradigms the thesis illustrates
in Figure 2 but never measured; they are included here as ablations.

All six answer **the profile's phrasing** of each question. The profile-blind
approaches simply never see the profile string. This is a deliberate change
from the original implementation — see `SV-03` in the errata.

### Experiments

| Config | What it does |
|---|---|
| `smoke` | 1 question × 6 approaches × 1 model. A few cents. Run this first. |
| `e1` | Effectiveness of personalization: 3 approaches × 7 models × 2 domains × 6 profiles × 2 difficulties = 504 cells. |
| `e1-ablations` | `e1` plus the three unmeasured paradigms, on 2 models. |
| `e2` | Variation between profiles: neutral vs. six personalized answers per question, compared with textual and semantic similarity. |

## Layout

```
configs/         domain · approach · model · judge · experiment  (the knobs)
data/            databases + questions.jsonl        (gitignored except corpus)
src/personareact/
  agent/         the LangGraph state machine: Reason, Act, Tailor
  tools/         Text-to-SQL over SQLite, with query auditing
  prompts/pt_BR/ prompts as files, scoped by domain, hashed into manifests
  corpus/        Question / Answer / Score records and jsonl io
  datasets/      database builders
  eval/          G-Eval judge, similarity metrics, statistics
  analysis/      tables and figures, as scripts
  runner.py      the sweep: resume, cost ceiling, manifests
results/<run>/   manifest.json · answers.jsonl · scores.jsonl
figures/ tables/ generated, never hand-edited
```

## Reproducibility

Every run writes `results/<run_id>/manifest.json` recording the git SHA, the
resolved config, the SHA-256 of every prompt file, the SHA-256 of the database,
the model snapshots, the sampling parameters and seed, the token counts and the
cost. A result can always be traced back to what produced it.

Databases are verified before every run: the checksum must match the one
declared in the domain config, and the expected SQL views must be present. This
check exists because the original tree had **three diverged builds** of the
soccer database spread across nine folders, and the treatment and the baseline
happened to use different ones (`SV-14` in the errata).

## Data availability

- **COVID-19** — public, from the FAPESP COVID-19 Data Sharing/BR repository
  (Hospital Sírio-Libanês subset). `docs/DATA.md` has the download and build
  procedure.
- **Football** — *not redistributable*. The tracking data was released under an
  agreement with Fortaleza Esporte Clube. The schema, the builders and the full
  question corpus are here, so the pipeline is auditable and runnable against a
  database with the same schema.

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE). The license covers the code in this repository,
not the datasets.

## Known cosmetic issue

Every command prints one `LangChainPendingDeprecationWarning` about
`allowed_objects` on startup. It comes from LangGraph 0.3.x, it is harmless,
and it survives `warnings.filterwarnings` because LangChain's deprecation
helper wraps the emission in its own `catch_warnings` block. Suppressing it
would mean redirecting stderr during import, which would hide real warnings
too, so it is left visible. It disappears on LangGraph 1.x, which is a
deliberate future change (see `docs/ERRATA.md`, "What is left").
