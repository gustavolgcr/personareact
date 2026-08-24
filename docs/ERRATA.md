# Errata

Defects found in the implementation used for the thesis *"PersonaReAct:
Personalização de Respostas em Modelos de Linguagem com base no Paradigma
ReAct"* (UFC, 2025), and what this repository does about each one.

Every entry names the file and line in the original tree, states the effect on
the published results, and links to the fix here. Nothing was quietly
corrected: the thesis artefacts are preserved unmodified under
`results/legacy-thesis-2025/` so the two can be compared.

Severity is about the **published results**, not the code:

| | |
|---|---|
| **Critical** | changes how a reported number should be read |
| **High** | affects internal validity or comparability between conditions |
| **Medium** | affects rigour, cost or reproducibility, not the direction of a result |

---

## Summary

| id | severity | what | status |
|---|---|---|---|
| [SV-01](#sv-01) | Critical | "Distância de Edição" is a *similarity*; the interpretation is inverted | fixed, quantified |
| [SV-02](#sv-02) | High | BLEU computed with hypothesis and reference swapped | fixed, quantified |
| [SV-03](#sv-03) | Critical | The ReAct baseline answered the neutral questions | fixed, needs regeneration |
| [SV-04](#sv-04) | Critical | The judge saw a different context for the baseline | fixed, needs re-judging |
| [SV-05](#sv-05) | High | One ReAct answer set scored six times, counted as six observations | fixed |
| [SV-06](#sv-06) | High | The medical case study ran with football prompts | fixed, needs regeneration |
| [SV-07](#sv-07) | High | The agent's chosen `action_input` was discarded | fixed |
| [SV-08](#sv-08) | High | The SQL model was coupled to the agent model | fixed (configurable) |
| [SV-09](#sv-09) | Medium | Failed runs were saved as answers and graded | fixed |
| [SV-10](#sv-10) | Medium | Nothing checked factual fidelity | fixed (new metric) |
| [SV-11](#sv-11) | Medium | Everything joined by list position | fixed |
| [SV-12](#sv-12) | Medium | No inferential statistics; outliers deleted | fixed |
| [SV-13](#sv-13) | Medium | Judge model unpinned (a floating alias) | fixed |
| [SV-14](#sv-14) | Critical | Treatment and baseline queried **different builds** of the database | fixed |
| [SV-15](#sv-15) | Critical | The summary SQL view inflated every aggregate, median **43×** | fixed |

Reproducibility and engineering findings (no runner, twelve repositories,
absolute paths in notebooks, no manifests, uninstallable packages, twelve `.env`
files) are addressed by the repository's existence and are not repeated here.

[`REVIEW.md`](REVIEW.md) assesses the implementation as *code* -- fidelity to
the formalisation in chapter 4 and the quality of the LangGraph usage -- and
covers findings that do not change a published number, such as the fact that
the outer Reason-Act loop iterates twice in 42 of 47 surviving traces.

---

## SV-01 — "Distância de Edição" is a similarity, read as a distance {#sv-01}

**Severity Critical.**

*Original:* `experiments/src/analysis/experiment_2/statistical_analysis_experiment_2_part_1.ipynb`,
cell 5. `from nltk.metrics.distance import edit_distance` is shadowed four
lines later by

```python
def edit_distance(a, b):
    return textdistance.levenshtein.normalized_similarity(a, b)
```

`normalized_similarity` returns `1 - lev/max(len)`: **high means similar.**

*Effect:* section 6.2 of the thesis reads the column as a distance — "ROUGE-L e
BLEU baixos e distância de edição alta indicam maior diferenciação" (p. 119),
"distância de edição alta como maior divergência" (p. 120). Both sentences are
inverted. Section 5.3 also defines the metric as the raw Levenshtein
recursion, which is not what was computed.

*Evidence.* Recomputing over the same answers separates the two quantities:

| soccer, overall | thesis "Distância de Edição" | `edit_similarity` | true `edit_distance` |
|---|---|---|---|
| o3-mini | 0.1544 | **0.1635** | 0.8365 |
| gpt-4.1-mini | 0.1966 | **0.2415** | 0.7585 |

The published values sit next to the similarity and nowhere near the distance.
Reproduce with `python scripts/compare_with_thesis.py`.

*Fix:* `personareact/eval/similarity.py` exposes `edit_similarity` and
`edit_distance_normalized` under unambiguous names, plus `levenshtein` for the
raw recursion. `tests/test_similarity.py` asserts the direction of each.

*What to change in the paper:* the two interpretation sentences. The ROUGE-L
and BLEU argument in section 6.2 is unaffected.

---

## SV-02 — BLEU with hypothesis and reference swapped {#sv-02}

**Severity High.**

*Original:* same notebook.

```python
def bleu_score(a, b):
    return sentence_bleu([a.split()], b.split(), smoothing_function=smoothie)
#                         ^ a is the REFERENCE list, b is the hypothesis
...
"bleu":    bleu_score(resposta, resposta_ref),    # personalized as REFERENCE
"rouge_l": rouge_l_score(resposta, resposta_ref), # personalized as hypothesis
```

*Effect:* BLEU and ROUGE used opposite conventions in the same table, and
BLEU's brevity penalty was applied to the neutral answer — systematically the
shorter of the pair, and shorter in a way that correlates with the effect being
measured. Section 5.3 states the neutral answer is always the reference.

*Evidence.* Same answers, corrected order:

| soccer, overall | thesis BLEU | corrected BLEU |
|---|---|---|
| o3-mini | 0.0088 | **0.0437** |
| gpt-4.1-mini | 0.0126 | **0.0855** |

Roughly a fivefold to sevenfold difference. The *ranking* of profiles is mostly
preserved, so section 6.2's qualitative conclusions survive; the absolute
values in Tables 13–18 do not.

*Fix:* every metric in `eval/similarity.py` takes `(hypothesis, reference)` in
that order, and `tests/test_similarity.py::TestArgumentOrder` fails if the
convention is ever reversed.

---

## SV-03 — The ReAct baseline answered the neutral questions {#sv-03}

**Severity Critical.**

*Original:* `react/src/main.py:56` sets `PROFILES = ["Neutral"]`, and the plain
ReAct answer files carry no profile in their names
(`answers_react_soccer_easy_gpt-4o-mini-2024-07-18.json`, every record with
`"profile": "Neutral"`). Meanwhile
`experiments/src/evaluations/evaluator_experiment_1.py:115` builds the judge's
input from the **profile-adapted** question and applies it to all three
conditions.

So for the same row of Table 1, PersonaReAct answered *"Qual foi a velocidade
média que o Bruno Pacheco alcançou naquele clássico contra o Ceará…"* while
ReAct answered *"Qual foi a velocidade média de Bruno Pacheco no jogo contra o
Ceará…"* — and the judge was shown the first.

*Effect:* the baseline differs from the treatments on two variables at once
(no profile in the prompt **and** a different question surface), so the
ReAct-versus-PersonaReAct gap in Tables 1–12 is not a clean contrast. Section
5.5 (p. 82) describes the opposite: "Cada pergunta, associada ao seu perfil, foi
processada pelas três abordagens".

*Fix:* `ExperimentConfig.cells()` expands every approach over every profile.
The profile-blind approaches receive the profile's phrasing of the question and
never see the profile string.
`tests/test_config.py::TestExperiments::test_every_approach_runs_on_every_profile`
enforces it.

*Status:* the code is fixed; the ReAct condition needs regenerating (84 cells
per domain) before the corrected numbers exist.

---

## SV-04 — The judge saw a different context for the baseline {#sv-04}

**Severity Critical.**

*Original:* `evaluator_experiment_1.py`, lines 129 / 134 / 139:

```python
caso_1 = LLMTestCase(..., context=["Perfil do usuário: " + profile])   # personareact
caso_2 = LLMTestCase(..., context=["Perfil do usuário: " + profile])   # react + prompt
caso_3 = LLMTestCase(..., context=["Sem perfil de usuário definido."]) # react
```

All three criteria list `LLMTestCaseParams.CONTEXT` in `evaluation_params`, so
the judge reads it.

*Effect:* the baseline was graded under a different prompt. When the criterion
is "does this style suit the profile" and the judge is simultaneously told
there is no profile, an unknown share of the measured gap belongs to the
harness. The gaps are large — on *Interpretação e Explicação dos Dados*,
PersonaReAct ≈ 9.0 against ReAct 3.5–5.8 — which is exactly the magnitude that
invites the question.

*Fix:* `eval/judge.py::build_test_case` constructs an identical `input` and
`context` for every condition. The approach name never enters the prompt, which
also blinds the judge.

*Status:* code fixed; re-judging the existing answers is cheap (no
regeneration) and is the recommended next step.

---

## SV-05 — One baseline answer set scored six times {#sv-05}

**Severity High.**

*Original:* `evaluator_experiment_1.py:88` — `chave_3 = ("react", dataset,
difficulty, model)` has no profile component, so the same file was re-fetched
inside the profile loop.

*Effect:* 12,600 judge calls per domain to score 140 distinct answers, about
10,500 of them duplicates; and in aggregation the ReAct column's *n* is
inflated sixfold with non-independent rows, so any test or interval computed
over it is wrong.

*Fix:* dissolved by SV-03 — the baseline is now genuinely per-profile.

---

## SV-06 — The medical case study ran with football prompts {#sv-06}

**Severity High.**

*Original:* one `prompts.py` served both domains.
`personareact/src/prompts/prompts.py:110-113` ends the Text-to-SQL prompt with

> IMPORTANT: A player can have multiple segments in the same match … you must
> GROUP all the segments of that player in the game and calculate the AVERAGE.

and lines 45–48 give the Tailor step four football personas (*Torcedor*,
*Narrador*, *Fisioterapeuta*) with bpm and m/s examples. `settings.py:5` points
at `covid.sqlite`. Git shows the football block present since commit `8aa992f`
(27 Jul 2025); every COVID answer was committed on 8 and 28 Aug 2025.

The same football prompt is reproduced as Appendix C of the thesis and
presented as *the* personalization prompt.

*Effect:* the medical agent was told to group patient records by "segments in a
match", and the medical Tailor step was shown football personas as its examples
of tone. Whatever the medical results measure, it is not a prompt designed for
the medical domain.

*Fix:* prompts live under `src/personareact/prompts/pt_BR/<domain>/`, resolved
from the domain config; per-profile tone guidance is data
(`profiles.yaml`) rather than prose baked into a template. A medical
Text-to-SQL prompt is written from the actual schema (patient/encounter keys,
`DE_ANALITO` filtering, the TEXT-typed `DE_RESULTADO`, birth year ≠ age).
`tests/test_prompts_and_agent.py::TestPromptScoping` fails if football
vocabulary appears in the medical prompt or vice versa.

*Status:* code fixed; the medical case study needs regenerating. It will likely
*improve* the reported numbers.

---

## SV-07 — The chosen `action_input` was discarded {#sv-07}

**Severity High.**

*Original:* `personareact/src/nodes/action_node.py:10`

```python
inp = state.question if state.action == "consulta_banco_sql" else state.action_input
```

`consulta_banco_sql` was the only tool ever registered, so the branch that
always fired threw away the Reason step's decision and re-sent the raw original
question.

*Effect:* the executed action was fixed rather than chosen, contradicting the
formalisation in section 4.1.6 where the policy π maps context to action; and
every iteration of the loop issued the same query, so a second iteration could
only return what the first did. Multi-step retrieval was impossible.

*Fix:* `agent/nodes.py::make_act_node` passes `action_input`.

---

## SV-08 — The SQL model was coupled to the agent model {#sv-08}

**Severity High.**

*Original:* `personareact/src/tools.py:16` builds a module-level `ChatOpenAI`
from the same global `OPENAI_MODEL`, then line 39 wraps it in a nested
`create_react_agent` over the SQL toolkit.

*Effect:* sweeping the model swept the SQL generator too, so H4 ("o tipo de
modelo influencia o desempenho das abordagens de personalização") cannot
separate personalization ability from text-to-SQL ability. A model that scores
lower may simply write worse SQL.

*Fix:* `ModelConfig.sql_model` is independent and recorded in the manifest.
Leaving it `null` reproduces the coupled behaviour; setting it holds retrieval
fixed while the personalization model varies.

*Not fixed, deliberately:* the tool is still internally a ReAct agent, so part
of the reasoning still happens outside the measured loop. Changing that would
break comparability with the thesis. It is a known limitation worth stating in
the paper.

---

## SV-09 — Failed runs saved as answers, then graded {#sv-09}

**Severity Medium.**

*Original:* `thought_node.py:36` raises `KeyError` when the structured output
omits `thought`; `main.py` catches it and writes `f"[ERRO: {e}]"` into the
answer field. The evaluator has no validity check.

*Effect:* 18 of the 3,640 answers on disk are not answers — 15 empty strings
and 3 error sentinels — and all were sent to G-Eval and scored. Small in
aggregate (0.5%), but three of them are in the ReAct column, where SV-05
replicates each row six times.

*Fix:* `LLMClient.structured` retries on both transport and shape failures;
`Agent.answer` marks empty answers **and tool errors that reached the answer
field** as `ok=False`; the judge skips them and records how many it skipped.

This one bit twice: the first smoke run of the rewrite marked an `act` cell as
successful because its answer *was* the tool's error string. Fixed by
`tools/text_to_sql.py::is_tool_error`, and covered by a test.

---

## SV-10 — Nothing checked factual fidelity {#sv-10}

**Severity Medium.**

*Original:* all three G-Eval criteria measure style, interpretation quality and
stylistic consistency. Nothing compares an answer to the database.

*Effect:* the strongest objection to a personalization layer — that it distorts
the data on the way out — could not be detected. The prompt explicitly tells
the model to "evite números crus" for some profiles. The artefacts show the
risk is real: a legacy answer reports "velocidade média … foi de 1.362 km/h"
for a column stored in metres per second, which Quadro 6 of the thesis renders
correctly as 1.36 m/s ≈ 4.9 km/h.

*Fix:* `eval/factuality.py` — every number stated in an answer must be
traceable, within a rounding tolerance, to a number the tool retrieved.
Optional gold values per question (`gold_sql` / `gold_answer`, see
`docs/GOLD.md`) add a second check. Computed programmatically, so it needs no
judge and costs nothing.

---

## SV-11 — Everything joined by list position {#sv-11}

**Severity Medium.**

*Original:* questions were bare JSON arrays of strings with no ids. The
per-profile rewrites came from `gpt-4o` at `temperature=0.7` with order
preservation requested only in the prompt text and never verified.
`respostas_1[i]`, `respostas_3[i]`, and a dataframe row index in the E2
notebook did the joining.

*Effect:* one dropped or reordered question anywhere would have silently
misaligned every comparison in the thesis. The migration verified the lengths
all match (280 questions, 364 answer files of exactly 10), so no drift
occurred — but nothing in the original could have detected it.

*Fix:* `data/questions.jsonl` with `question_id` and `neutral_id`; every answer
and score record carries `question_id`; joins are merges on a key and
`Corpus` raises on duplicate or dangling ids. The migration asserts alignment
and aborts rather than shifting.

---

## SV-12 — No inferential statistics, and outliers deleted {#sv-12}

**Severity Medium.**

*Original:* four notebooks compute means and standard deviations. `scipy` is a
declared dependency and never imported. An IQR filter replaces out-of-range
G-Eval scores with `NaN` before aggregation, and the aggregation groups by
criterion alone — pooling 7 models × 6 profiles × 2 difficulties × 20 questions
× 5 repetitions into a single mean and a single "DP".

*Effect:* the reported standard deviation mixes judge repetition noise with
between-question and between-profile variance and answers no question. H2, H3
and H4 rest on differences of 0.01–0.2 points on a 0–10 scale with no test
attached, and the outlier filter removes precisely the disagreements that would
show up as variance.

*Fix:* `eval/stats.py` — repetitions collapse to one observation per answer,
comparisons are paired on `question_id`, significance from Wilcoxon
signed-rank with Holm correction, effect size as rank-biserial correlation,
plus a bootstrap CI on the mean paired difference. No outlier removal.

---

## SV-13 — The judge model was a floating alias {#sv-13}

**Severity Medium.**

*Original:* `criteria.py` carries `# model=OPENAI_MODEL` commented out on every
metric, so deepeval fell back to its own default. In the vendored deepeval
2.9.7, `models/llms/openai_model.py:141` sets `default_gpt_model = "gpt-4o"`.

*Effect:* resolves the contradiction between section 5.3 (GPT-4o) and section
5.5 (4o-mini) in favour of 5.3 — but nothing in the repository recorded it, and
`gpt-4o` is repointed by OpenAI over time, so the published scores are not
recoverable.

*Fix:* the judge config names `gpt-4o-2024-11-20`, `JudgeConfig.load` **rejects**
a known floating alias, and the snapshot is written into every judge manifest.

---

## SV-14 — Treatment and baseline queried different databases {#sv-14}

**Severity Critical.** *Found during the audit; not visible from the code alone.*

The football database had been copied into nine project folders. Checksums show
three distinct builds:

```
9439e0ec  2025-05-09  react, react_prompt_personalization,
                      questions-generator-llm-soccer, act, data-processing
426f70f7  2025-04-24  personareact, cot, standard
73786dae  2025-05-08  (backup)
```

Row counts are identical in all three. The schemas are not: the 09 May build
carries `View_EstatisticasComJogadorJogo` and `View_ResumoPorJogadorEPartida`;
the 24 April build has no views at all. Both
`personareact/src/tools.py:17` and `react/src/tools.py:18` pass
`view_support=True`, so the views enter the schema handed to the Text-to-SQL
agent.

*Effect:* during the August 2025 runs, the ReAct and ReAct-enriched baselines
queried a database offering a ready-made per-player-per-match aggregate, while
PersonaReAct queried one where it had to write the JOIN and GROUP BY itself
over 164,618 segment rows. Retrieval difficulty was not held constant between
conditions. PersonaReAct's copy was simply never updated when the views were
added on 09 May.

*Fix:* one database per domain at the path its config declares, its SHA-256
pinned in `configs/domain/*.yaml` and verified before every run, along with the
expected view list. A mismatch aborts the run.

---

## SV-15 — The summary view inflated every aggregate {#sv-15}

**Severity Critical.** *Found during the audit.*

`View_ResumoPorJogadorEPartida` (defined in
`data-processing/src/create_database.py:228`) joins `Pontos_Trajetoria` to
`Estatisticas_Segmento` **before** aggregating, fanning each segment out to one
row per trajectory point. Segment lengths vary from 3 to 30+ points.

*Effect:* every column of that view is wrong.

| Brayan vs Ceará, 2023-02-07 | view | truth |
|---|---|---|
| `Total_Sprints` | 2,220 | 62 |
| `Tempo_Total_Parado` | 12,226 | 974 |
| `Velocidade_Media` | 1.4629 | 1.0264 |

Across the 727 player-match rows the `SUM` columns are inflated by a **median
factor of 43×** (range 3× to 107×). The `AVG` columns are point-weighted
instead of segment-weighted, so they are wrong too.

Combined with SV-14, this means the baselines had access to — and the prompt
steered them toward — a view returning numbers wrong by up to two orders of
magnitude, while PersonaReAct computed its own aggregates from the raw tables.
The value in Quadro 6 of the thesis (1.3622634444889454 m/s for Bruno Pacheco
vs Ceará on 2023-04-01) matches the *correct* aggregation exactly, which is
consistent with PersonaReAct having no access to the broken view.

*Fix:* `datasets/soccer.py::VIEWS` aggregates the segment statistics directly
and pulls total distance from a scalar subquery. `personareact data
import-legacy` replaces the views on an existing database — views are metadata,
so no data is touched and no rebuild is needed. The corrected view now matches
a direct aggregation exactly.

---

## Documentation-only discrepancies

Not defects in the code, but places where the thesis text and the
implementation disagree. Worth resolving before the paper repeats them.

| Thesis | Implementation | Note |
|---|---|---|
| §5.5 p. 85: embeddings from `all-MiniLM-L6-v2`, with a paragraph justifying the choice | the notebook uses `all-mpnet-base-v2` | §5.3 p. 78 says mpnet and is correct |
| §5.5 p. 83: judge is `4o-mini` | deepeval default `gpt-4o` | §5.3 p. 76 says GPT-4o and is correct (SV-13) |
| §5.5 p. 85: the E2 neutral answer is "PersonaReAct com o módulo Tailor desativado" | no such switch existed; the notebooks used the plain-ReAct files | now genuinely available: run `personareact` with `--profile neutral` |
| §5.1 p. 72: "20 perguntas em tom neutro", then "cada uma das **10** perguntas neutras" | 10 easy + 10 hard per domain, each rewritten for 6 profiles | the 140 total is right; the difficulty split is never mentioned or analysed |
| §5.2 p. 75: after 5 iterations "o sistema emite uma resposta padrão de falha" | control passes to the Tailor step, which personalizes whatever is in the scratchpad | no failure message exists; truncation is now flagged on the record |
| §4.4: "se nenhum perfil for fornecido, o modelo pode ser simplificado para o ReAct tradicional" | `profile` was a required field and the Tailor node always ran | implemented here; `AgentState.profile` is optional |
| §5.2 p. 75: agents used GPT-4o mini, temperature 1 | E1 sweeps seven models; temperature implicit in two of three call sites | now explicit per model config |
| §5.5 p. 83: `top_p = 1.0`, `frequency_penalty = 0.0` | never sent; these are API defaults | now sent explicitly |
| p. 74 footnote 3: `github.com/gustavolgcr/personareact` | HTTP 404; the remotes were `*-phd` | this repository is intended for that URL |
| Appendix B, Código-fonte 1 | criterion text reads "esse tipo de perfol" | typo fixed in `eval/criteria.py` |

---

## What is left

Ordered by how much it changes.

1. **Re-judge the existing answers** with the symmetric harness (SV-04, SV-13)
   and the failure exclusion (SV-09). No regeneration, judge calls only. This
   is the cheapest way to learn whether the headline results survive.
2. **Regenerate the ReAct baseline** on profile-adapted questions (SV-03).
3. **Regenerate the medical case study** with domain-correct prompts (SV-06),
   against the corrected database (SV-14, SV-15).
4. **Author gold answers** for the 40 neutral questions so the factuality check
   has ground truth (`docs/GOLD.md`).
5. **Report difficulty as a factor** — it is in every record and was never
   analysed.
6. **Move to LangGraph 1.x**, which removes the startup warning and the pinned
   0.3.x stack. Deferred because it changes agent behaviour and would confound
   the comparison with the thesis.
