# Corrected implementation vs. the thesis implementation

A controlled head-to-head, everything held constant except how the agent is
built: same questions, same models, same profiles, same database, same judge,
same harness.

```bash
personareact run     --experiment compare
personareact judge   --run <run_id> --repetitions 3
personareact analyze --run <judge_run_id>
```

**Scope.** 24 cells, 120 answers, 0 failures, US$ 1.47 to generate and about
US$ 3 to judge. Soccer, easy, 5 questions, two models (o3-mini with reasoning,
gpt-4.1-mini without), two profiles (Torcedor, expressive; Técnico, technical),
three approaches in both their corrected and `-thesis` forms.

**This is a directional signal, not a settled result.** Twenty answers per cell
and three judge repetitions. `e1` against `e1-thesis` is the same design at
full scale and would settle it.

---

## Generation: better on every axis at once

| approach | variant | iterations | tokens | USD | numbers grounded | answers with no data at all |
|---|---|---|---|---|---|---|
| personareact | **corrected** | 4.05 | 18,968 | 0.011 | **94.1%** | 15% |
| personareact | thesis | 2.00 | 51,799 | 0.018 | 87.5% | 20% |
| react | **corrected** | 4.20 | 13,214 | 0.008 | **100%** | 0% |
| react | thesis | 2.00 | 42,258 | 0.017 | 94.1% | 15% |
| react_enriched | **corrected** | 4.20 | 14,648 | 0.010 | **100%** | 5% |
| react_enriched | thesis | 1.75 | 32,825 | 0.010 | **70.6%** | 15% |

Three things improve together, which is unusual:

* **the loop actually is ReAct** — 4.05 iterations against 2.00, with a real
  list → describe → query → answer trace;
* **it costs less, not more** — about 2.7x fewer tokens and ~40% lower cost on
  average. Removing the nested model more than pays for the extra iterations;
* **factual grounding rises** in all three approaches. The starkest case is the
  thesis' `react_enriched`: **29.4% of the numbers it states are not traceable**
  to anything the tool returned. Corrected, that is 0%.

## Judgement: H1 strengthens, H3 goes from refuted to tied

Paired Wilcoxon on the question, Holm-corrected within each family, rank-biserial
effect sizes.

**H1 — PersonaReAct vs ReAct.** Holds in both, and the corrected version is
stronger:

| criterion | corrected | thesis |
|---|---|---|
| data interpretation | **+2.95** (large, p<0.001) | +2.71 (large, p<0.001) |
| language and style | **+3.63** (large, p<0.001) | +3.07 (large, p<0.001) |
| style consistency | **+2.06** (large, p=0.001) | +1.82 (large, p<0.001) |

**H3 — PersonaReAct vs ReAct + enriched prompt.** This is where the fixes
matter:

| criterion | corrected | thesis |
|---|---|---|
| data interpretation | −0.19, p=0.95, **not significant** | **−2.06, p=0.001, large — PersonaReAct significantly worse** |
| language and style | +0.73, p=0.74, not significant | −0.72, p=0.23, not significant |
| style consistency | +0.07, p=0.95, negligible | −0.27, p=0.23, small |

Under the thesis implementation the enriched prompt **beats** PersonaReAct on
data interpretation by two full points with a large effect. Under the corrected
implementation the two are statistically indistinguishable on all three
criteria, and PersonaReAct is nominally ahead on language.

So the fixes do not make H3 true. They rescue it from being refuted.

## What this means for the paper

**H1 is safe and gets stronger.** Personalization beats no personalization,
with large effects, in both implementations.

**H3 cannot be claimed as stated.** "A separação explícita da personalização
melhora a organização das respostas" is not supported: at full scale on the
legacy scores it holds in only 22 of 42 comparisons, and here PersonaReAct and
the enriched prompt tie.

**But the tie is the interesting result**, once grounding is on the table.
Corrected PersonaReAct grounds 94.1% of the numbers it states; the thesis'
enriched prompt grounds 70.6%. A modular personalization step that matches
prompt enrichment on perceived quality *while being substantially more faithful
to the data* is a better claim than a marginal win on a style rubric — and it is
one the thesis' evaluation had no instrument to make.

It also exposes a limit of the judge. The thesis' `react_enriched` scored
**8.76** on "Interpretação e Explicação dos Dados" while inventing or converting
three of every ten numbers it stated. G-Eval rewards fluent interpretation, not
correct interpretation. That is the gap `eval/factuality.py` exists to close.
