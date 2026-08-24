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

**The judge was the oldest thing in the pipeline — now switched.** It had been
pinned to `gpt-4o-2024-11-20`: a 2024 model grading 2026 answers, and one that
`docs/COMPARISON.md` shows giving 8.76 on "Interpretação e Explicação dos
Dados" to a condition that invented or converted three numbers in ten. A judge
that scores fluent interpretation as correct interpretation is the wrong
instrument for this thesis.

The default is now `geval-gpt-5.6-sol`. `geval-gpt4o-2024-11-20` stays, because
every published score came from it and it is the only way to make a new run
comparable with the tables in chapter 6. `compare` and `e1-thesis` deliberately
still use it; everything else moved.

**Scores from the two judges are not comparable.** Re-judge a whole run with
one or the other; never mix them in one table.

### What switching the judge actually required

Not a one-line config change. deepeval 2.9.7 predates every model released
since and carries three hardcoded registries, each failing differently:

| registry | failure when a model is missing |
|---|---|
| `valid_gpt_models` | `ValueError: Invalid model`, immediately |
| `structured_outputs_models` | **silent** — G-Eval drops off the structured-output path and rubric scoring degrades to free-text parsing |
| `model_pricing` | `KeyError: 'input'`, *after* the call has been paid for |

The middle one is the dangerous one, and it is why
`eval/criteria.py::_register_with_deepeval` registers all three rather than
just the price.

There is a second trap. deepeval passes `temperature` on **every** completion
call and defaults it to `0`, which reasoning models reject. So the judge is now
built as a configured `GPTModel` instance rather than handed to G-Eval as a
model name, and `temperature` is a field on the judge config: `1.0` for
gpt-5.6-sol, `0.0` for the legacy gpt-4o.

One limitation left standing: deepeval 2.9.7 has no way to pass
`reasoning.effort`, so the judge runs at the model's default. Raising it would
need a `DeepEvalBaseLLM` subclass.

**Untested against the live API.** The switch was made without running
anything, so the temperature interaction is verified only against deepeval's
call signature, not against OpenAI's response. Check it for a few cents before
committing to a full sweep:

```bash
personareact judge --run <a small run> --limit 2 --repetitions 1
```

## Beyond OpenAI

Section 5.6 lists single-vendor dependence as a limitation: "o uso restrito a
uma única fornecedora impede avaliar possíveis variações de desempenho e
compatibilidade em modelos de outros fabricantes".

Adding a second vendor is now a config file plus one change in code: `llm.py`
constructs `ChatOpenAI` directly, so it would need to dispatch on a `provider`
field to the matching LangChain chat class. The approach, prompt, corpus,
judge and analysis layers are provider-agnostic already. That is the cheapest
remaining way to close a limitation the thesis itself names.
