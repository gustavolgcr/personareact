# Choosing models, August 2026

The thesis used seven OpenAI models chosen in 2025. **Five of the seven are
scheduled for shutdown within four months of this writing**, so this is not an
optional refresh.

## The deprecation problem

| Thesis model | Status |
|---|---|
| `o3-mini-2025-01-31` | **retires 2026-10-23** |
| `o4-mini-2025-04-16` | **retires 2026-10-23** |
| `gpt-4.1-nano-2025-04-14` | **retires 2026-10-23** |
| `gpt-5-mini-2025-08-07` | **retires 2026-12-11** |
| `gpt-5-nano-2025-08-07` | **retires 2026-12-11** |
| `gpt-4.1-mini-2025-04-14` | available |
| `gpt-4o-mini-2024-07-18` | available |

After those dates the API returns errors for those ids. Two consequences:

1. **Chapter 6 stops being reproducible.** The answers and scores are archived
   in `results/legacy-thesis-2025/`, and the analysis regenerates from them --
   but nobody will be able to *re-generate* them. Say so in the paper and point
   at the archive.
2. **`e1` and `e1-thesis` have a deadline.** If the full-scale corrected-vs-
   thesis comparison matters, it has to run before 23 October.

## The axis H4 was about no longer exists as a model property

The thesis split the seven into "com raciocínio" (o3-mini, o4-mini, GPT-5
mini/nano) and "sem raciocínio" (4o-mini, GPT-4.1 mini/nano). That split
confounded four things at once: reasoning, model family, parameter count and
training generation.

In the GPT-5.6 generation reasoning is a **parameter**, not an identity:
`reasoning.effort` takes `none | low | medium | high | xhigh | max` on the same
model. So H4 becomes a controlled experiment instead of an observational one.

A single-question probe on `gpt-5.6-luna`, same prompt, same database, same
approach — only the dial moves:

| effort | iterations | tool calls | tokens | USD | outcome |
|---|---|---|---|---|---|
| `none` | 1 | **0** | 1,827 | 0.0005 | declines: *"Melhor não inventar um número"* |
| `medium` | 4 | 3 | 8,000 | 0.0023 | grounded: *"aproximadamente 1,36 m/s"* |
| `high` | 4 | 3 | 7,977 | 0.0026 | grounded |

At `none` the agent **does not act at all**. So the dial does not merely change
answer quality — it decides whether the Reason–Act cycle happens. That is a
sharper version of H4 than the thesis could ask, and `configs/experiment/h4-effort.yaml`
runs it for about US$ 1.

## Recommended lineup

| config | model | in / out per 1M | role |
|---|---|---|---|
| `gpt-5.6-luna` | `gpt-5.6-luna` | $0.20 / $1.20 | the volume tier; replaces the nano/mini role |
| `gpt-5.6-terra` | `gpt-5.6-terra` | $2.00 / $12.00 | mid tier |
| `gpt-5.6-sol` | `gpt-5.6-sol` | $4.00 / $20.00 | flagship ceiling reference (promotional through 2026-11-21; list is $5/$30) |
| `gpt-4.1-mini` | dated snapshot | $0.40 / $1.60 | **bridge to the thesis** — survives the cull |
| `gpt-4o-mini` | dated snapshot | $0.15 / $0.60 | **bridge to the thesis** — survives the cull |

Keeping the two survivors matters: they are the only models that appear in both
the published tables and any new run, so they anchor the comparison.

`configs/experiment/e1-2026.yaml` sweeps exactly this set.

### Cost is no longer the constraint it was

Section 5.6 justifies the mini/nano restriction on cost: "o uso de modelos mais
robustos [seria] financeiramente inviável". That has changed twice over.

Luna is cheaper than the models the thesis called affordable, and the corrected
pipeline removed the nested SQL agent, cutting token use about 2.7x. Measured
on one question, PersonaReAct end to end:

| model | tokens | USD |
|---|---|---|
| `gpt-4.1-mini` (corrected) | 6,897 | 0.0035 |
| `gpt-5.6-luna` (corrected) | 8,000 | **0.0023** |
| `gpt-4.1-mini` (thesis impl.) | ~24,000 | ~0.012 |

A full 504-cell Experiment 1 sweep costs roughly **US$ 1.40 per model on Luna**
and about US$ 26 on Sol. The flagship tier is now affordable for a thesis-scale
experiment, which removes a stated limitation rather than restating it.

## Two things that got worse

**No dated snapshots.** OpenAI stopped publishing them with the 5.6 family:
`gpt-5.6-luna` *is* the snapshot name. The "pin a dated snapshot" rule behind
`SV-13` cannot be satisfied, so the configs declare `no_dated_snapshot: true`
and a test enforces that the admission is explicit. Mitigation: every run
manifest records the date, and judge runs archive each raw verdict, so a later
drift is at least detectable.

**The judge is now the oldest thing in the pipeline.** It is pinned to
`gpt-4o-2024-11-20`, which is sound for comparability with the legacy scores
but is a 2024 model grading 2026 answers — and the comparison in
`docs/COMPARISON.md` shows it giving 8.76 on "Interpretação e Explicação dos
Dados" to a condition that invented three numbers in ten. If you re-judge
everything anyway, `gpt-5.6-terra` is the better instrument; if you want to
compare against the legacy scores, it has to stay.

## Beyond OpenAI

Section 5.6 lists single-vendor dependence as a limitation: "o uso restrito a
uma única fornecedora impede avaliar possíveis variações de desempenho e
compatibilidade em modelos de outros fabricantes".

Adding a second vendor is now a config file plus one change in code: `llm.py`
constructs `ChatOpenAI` directly, so it would need to dispatch on a `provider`
field to the matching LangChain chat class. The approach, prompt, corpus,
judge and analysis layers are provider-agnostic already. That is the cheapest
remaining way to close a limitation the thesis itself names.
