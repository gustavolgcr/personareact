# Review of the original implementation

An assessment of the LangChain/LangGraph code used for the thesis, on two
questions:

1. **Fidelity** — does the code do what chapters 4 and 5 say it does?
2. **Correctness** — is the LangGraph/LangChain usage sound and idiomatic?

The short version: **the architecture is right and the graph matches Figure 5
exactly.** What is wrong is mostly at the seams — how the Reason step's
decision reaches the tool, how much of the context reaches the Tailor step, and
a handful of LangChain usages that work today but are fragile or wasteful. One
finding is structural and worth thinking about before the paper: in practice
the outer Reason–Act loop almost never iterates.

Defects that affect the published numbers are in [`ERRATA.md`](ERRATA.md); this
document is about the code as code.

---

## 1. What is genuinely right

**The graph topology matches the thesis.** Figure 5 shows `__start__ →
thought_node`, a conditional fan-out to `action_node` / `persona_node` / itself,
`action_node → thought_node`, and `persona_node → __end__`. The code builds
exactly that. The self-loop fallback, the dashed conditional edges, the
placement of `persona_node` as the sole terminal — all correct.

**The state machine is the right abstraction** for this problem, and using
`StateGraph` directly rather than a prebuilt agent is the right call: it is
what makes the Tailor step a *separable module* rather than another instruction
in a prompt, which is the thesis' actual contribution.

**The iteration cap works as described.** `iteration_count >= 5` counts Reason
steps, and section 5.2's "limitado a 5 ciclos" is accurate.

**Structured output instead of parsing free text.** `with_structured_output`
over a `TypedDict` is a far more robust way to get Thought/Action/Answer than
regex-parsing a completion, which is what the original ReAct paper had to do.
This is a genuine improvement over the 2023 approach, not a deviation.

---

## 2. Fidelity gaps

### 2.1 The Tailor receives the answer it is supposed to be composing

**This is the most consequential finding in this document.**

Section 4.2 formalises the personalization function as

> τ : P × O* → M

— the profile and **O\***, "a sequência de observações obtidas ao longo dos
passos *Act*". Section 4.4 builds on it: "a dissociação entre o que o agente
observa e o que é efetivamente comunicado ao usuário".

`persona_node.py` passes the whole scratchpad:

```python
history = "\n".join(state.scratchpad)
```

and the scratchpad, by the time the Tailor runs, contains the thoughts, the
actions, the observations **and** the line `Standard Answer: ...`. From a real
trace in `personareact/src/logs/covid/individual_log.log`:

```
Question: Entre janeiro e junho de 2020, qual foi o resultado mais frequente…
Thought: Preciso consultar o banco de dados…
Action: consulta_banco_sql
Action Input: Qual foi o resultado mais frequente nos hemogramas…
Observation: {"Desfecho mais frequente": "Hemograma com resultado normal"}
Thought: A partir da observação, posso concluir a resposta.
Standard Answer: O resultado mais frequente nos hemogramas… foi 'Hemograma…'
Personalized: Nos meses de janeiro a junho de 2020, a maioria dos hemogramas…
```

So τ is implemented as τ(p, c_t) — profile and *full context* — not τ(p, O\*).
The Tailor is handed a finished answer to rewrite. That is a weaker claim than
the thesis makes: the neutral answer's wording, framing and choice of which
facts to include all anchor the personalized one.

**Fixed.** `tailor_sees: observations` is now the default for `personareact`,
so the implementation matches the formalisation. `personareact-thesis` keeps
`context` for reproduction.

The first attempt at this exposed a second problem: with only observations, the
Tailor produced *"cerca de 1,36 (naquela unidade de medida do time)"* -- the
number survived but the unit did not, because the nested agent returned a bare
`{"Velocidade_Media": 1.3622634444889454}` and the unit had only ever existed
in the standard answer. The real fix was upstream: the SQL tools now annotate
result columns from a unit map declared in the domain config, so an observation
reads `{"Velocidade_Media (m/s)": 1.36}` and the Tailor answers *"cerca de 1,36
metros por segundo"*. The unit travels with the datum.

The two modes still diverge, which is what makes the ablation worth running:

| mode | answer, on the same question |
|---|---|
| `context` (thesis) | "…mantendo uma velocidade média daquelas que mostram raça e entrega — não chega a ser um foguete…" — **no number at all** |
| `observations`, before the unit fix | "…a média registrada foi de cerca de 1,36 (naquela unidade de medida do time)…" — datum kept, unit lost |
| `observations` (current default) | "…Foi cerca de 1,36 metros por segundo, o que pra um jogo daquela intensidade mostra como ele se dedicou…" — **datum and unit both preserved, in the profile's voice** |

The `context` mode, anchored by a finished answer, drops the figure entirely.
The `observations` mode composes from the retrieved facts and keeps it. Which
is better depends on the profile and on how much you value information
preservation -- which is exactly why it deserves to be an experiment rather
than an accident.

### 2.2 The Reason–Act loop almost never iterates

From the surviving execution logs (47 traces — the logs were untracked and
appended per run, so this is a sample, not the full 3,640):

| Reason iterations | answers |
|---|---|
| 2 | 42 |
| 3 | 2 |
| 5 (the cap, i.e. truncated) | 3 |

Two iterations means: reason → act → reason → answer. **One tool call, one
reasoning step on each side of it.** In ~89% of the sample, the interleaving of
reasoning and acting that ReAct is *about* does not happen in the measured
loop.

**Fixed.** The default `toolset` is now `primitives`: `listar_tabelas`,
`descrever_tabelas` and `executar_sql` are exposed directly to the outer loop.
Measured on the same question, the loop went from 2 iterations to 4-5, with a
real trace:

```
Thought: preciso listar as tabelas disponíveis
Action: listar_tabelas      -> Tabelas: Estatisticas_Segmento, Jogadores, ...
Thought: vou descrever a view View_ResumoPorJogadorEPartida
Action: descrever_tabelas   -> CREATE VIEW View_ResumoPorJogadorEPartida AS ...
Thought: agora posso consultar
Action: executar_sql        -> [{"Velocidade_Media (m/s)": 1.3622634444889454}]
Thought: já tenho o dado
```

Two further consequences, both good: there is **no second language model** at
all, which removes the `SV-08` confound rather than making it configurable; and
a SQL error becomes an observation the agent corrects on the next iteration,
which is the behaviour ReAct exists to produce.

`max_iterations` rises from 5 to 8, because primitives need more steps for the
same work. The `-thesis` variants keep 5 and the nested agent.

The original causes, for the record:

- the tool is itself a complete ReAct agent (`create_react_agent` over
  `SQLDatabaseToolkit`), so it lists tables, reads schemas, writes SQL, reads
  results and retries internally. By the time it returns, there is nothing left
  for the outer loop to do;
- `action_node.py:10` discarded the Reason step's `action_input` and re-sent the
  original question, so a *second* tool call would have returned exactly the
  same observation. The model correctly concludes it should stop
  (ERRATA `SV-07`).

The consequence for the paper: what E1 compares is not "ReAct with N cycles"
versus "PersonaReAct with N cycles". It is a thin reasoning wrapper around a
Text-to-SQL agent, plus nothing, plus a profile suffix, or plus a Tailor step.
The differences between the three conditions live entirely in the presentation
layer.

That is not fatal — PersonaReAct *is* about the presentation layer, and section
4.2 says so explicitly ("não altera o processo de decisão interna do agente,
mas atua na etapa final"). But Figure 5 and section 5.2 present the outer loop
as the system under study, and a reviewer who instruments it will find that it
barely runs. Better to state it: report the iteration distribution, and frame
the contribution as the Tailor step over a fixed retrieval backend.

### 2.3 Claims in the thesis with no implementation behind them

| Claim | Reality |
|---|---|
| §4.4 "se nenhum perfil for fornecido, o modelo pode ser simplificado para o ReAct tradicional" | `profile` was a required field of `PersonaReActState` and `persona_node` always ran. No profile-less path existed. |
| §5.2 "após o número máximo de iterações… o sistema emite uma resposta padrão de falha" | Control passes to `persona_node`, which personalizes whatever is in the scratchpad. There is no failure message, and a truncated run is indistinguishable from a complete one in the output file. |

Both are implemented in the rewrite (`AgentState.profile` is optional;
truncation is flagged on the record).

---

## 3. LangGraph and LangChain usage

Ordered by how much it matters.

### 3.1 A third-party HTTP call on every single run

```python
# react/src/main.py:42  — module level, outside `if __name__ == "__main__"`
react_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

`draw_mermaid_png` defaults to `draw_method=MermaidDrawMethod.API`, which
**POSTs the graph to mermaid.ink**, a public third-party web service. Because
the call sits at module level, it fired on every invocation of the ReAct
baseline — the script that produced *both* baseline conditions — before a
single question was answered. If mermaid.ink was slow, rate-limiting or down,
the run died before doing any work.

The same line is in `cot/src/main.py:15` and `standard/src/main.py:16`. It is
commented out in `personareact` and `act`, which is another silent asymmetry
between conditions.

Diagram generation belongs behind a flag, and should use
`draw_method=MermaidDrawMethod.PYPPETEER` or just `draw_mermaid()` (which
returns text and needs no network).

### 3.2 Callables stored in the graph state

```python
@dataclass
class PersonaReActState:
    tools: Dict[str, Callable[[str], str]]
```

Function objects in the state make it unserialisable, which rules out
checkpointers, `astream`, time-travel and resumption — all the reasons to use
LangGraph rather than a `while` loop. It is also unnecessary: nodes can close
over their dependencies, which is what the rewrite does.

### 3.3 Module-level side effects, and rebuilding the agent per call

```python
# tools.py, at import time
llm = ChatOpenAI(model=OPENAI_MODEL, openai_api_key=OPENAI_API_KEY)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}", view_support=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
prompt = text_to_sql_prompt.format(dialect="SQLite")
```

Importing the module opens the database and pins the model. Changing
`settings.OPENAI_MODEL` afterwards has no effect, and testing anything that
imports `tools` requires a live database.

Then, inside the tool:

```python
agent_executor = create_react_agent(llm, tools, prompt=prompt)
```

The nested agent is **rebuilt on every single tool call** — 3,640 times across
the experiments — instead of once per process.

### 3.4 A loop variable used after the loop

```python
for step in agent_executor.stream({...}, stream_mode="values"):
    step["messages"][-1].pretty_print()
return step["messages"][-1].content
```

If the stream yields nothing, `step` is undefined and this raises `NameError`,
which the caller turns into a `[Erro ao executar consulta]` string that flows
onward as if it were data. `stream_mode="values"` also yields the *entire*
message list at every step, so `pretty_print()` dumps the whole growing
conversation to stdout once per step, for every question.

`.invoke()` is the right call here; streaming buys nothing when the result is
consumed all at once.

### 3.5 `with_structured_output` without `include_raw`

Three consequences:

- **no usage metadata**, so token counts and cost cannot be measured at all —
  which is why the thesis reports no cost figures;
- **no way to distinguish a parse failure** from a model that legitimately
  omitted an optional key;
- `result['thought']` raises `KeyError` when the key is absent, and the
  top-level handler writes the literal string `[ERRO: 'thought']` into the
  answer field (ERRATA `SV-09`).

`include_raw=True` returns `{"raw", "parsed", "parsing_error"}` and solves all
three.

### 3.6 Key-presence used as a completion signal

```python
if "standard_answer" in result:
```

With `TypedDict(total=False)` and structured output, a model that returns
`{"thought": "...", "standard_answer": ""}` satisfies `in` and ends the loop
with an empty answer. Truthiness is the correct test. Several of the 15 empty
answers in the corpus are consistent with this path.

### 3.7 Tools invoked as plain callables

```python
tool_fn = state.tools.get(state.action)
obs = tool_fn(inp)
```

`@tool` returns a `StructuredTool`, and `BaseTool.__call__` has been deprecated
since langchain-core 0.1.47 **and is removed in 1.0**. It works on 0.3.55 and
emits a `LangChainDeprecationWarning`; it will stop working on any upgrade.
`tool_fn.invoke(inp)` is the supported form.

### 3.8 Smaller things

- **A side effect in a routing function.** `decide_condition` prints. Routers
  are called by the framework and should be pure.
- **Inconsistent sampling between the paradigms.** `cot` and `standard` set
  `temperature=0`; `react` and `personareact` set nothing (API default 1.0);
  `persona_node` sets `1` explicitly. The paradigms are not comparable to each
  other on sampling, which matters if the Figure 2 baselines are ever measured.
- **No reducers on the state.** Without `Annotated[list, operator.add]`, every
  node must copy and return the whole scratchpad. It works, but the pattern
  used — assigning `updates["scratchpad"] = new_scratch` *before* appending
  more items to `new_scratch` — depends on the two names aliasing the same
  list. Correct today, silently wrong the moment someone adds a `.copy()`.
- **No `recursion_limit` set.** The default is 25 and the worst case here is
  about 11 steps, so there is headroom — but it is accidental headroom, not
  chosen.
- **No retries anywhere.** A single transient 429 or timeout became an `[ERRO:
  ...]` string in the results file.

---

## 4. Fidelity to ReAct (Yao et al., 2023)

The original ReAct is a *prompting* method: few-shot exemplars containing
interleaved `Thought: … / Action: … / Observation: …` traces, with a
`finish[answer]` action in the action space.

This implementation is a zero-shot, structured-output reimplementation with the
same control flow. That is a reasonable modernisation and the thesis discloses
it ("utilizando um prompt adaptado para este projeto", §5.2). Two things worth
saying out loud in the paper:

- the ReAct baseline is a **reimplementation**, not the original prompting
  scheme, so absolute comparisons with published ReAct numbers are not
  meaningful (relative comparisons within this study are fine — all three
  conditions share the same implementation);
- because the tool is itself an agent, the outer loop is much shallower than
  ReAct's, per §2.2 above.

---

## 5. What was changed in the rewrite, and what was left alone

**Fixed:** the nested agent replaced by primitives, so the Reason-Act cycle
actually interleaves and no hidden second model is involved; the Tailor input
narrowed to O* with units carried in the observation; the discarded
`action_input` (including the subtler case where an empty argument was replaced
by the question -- legitimate for `listar_tabelas`); the nested agent built once
instead of per call; `include_raw` plus retries; truthiness instead of key presence;
`invoke()` instead of `__call__`; no callables in state; no module-level side
effects; no network call to draw a diagram; explicit sampling parameters for
every approach from one config; the Tailor input made explicit and selectable.

**Left alone, deliberately:** the tool is still internally a ReAct agent. Fixing
that would change what the system does and break comparability with every
number in the thesis. It is a limitation to state in the paper, not a bug to
patch silently.

## 6. If you only do three things

1. **Report the iteration distribution** and frame the contribution as the
   Tailor step over a fixed retrieval backend. It is a smaller claim than
   "PersonaReAct extends the ReAct cycle", and it is the one the data supports.
2. **Run the `personareact-obs` ablation.** The gap between §4.2's τ(p, O\*)
   and the implemented τ(p, c_t) is real, observable and cheap to measure — and
   it goes straight to whether personalization preserves information.
3. **Report cost.** PersonaReAct adds one LLM call per question over ReAct. The
   measured overhead is small relative to the Text-to-SQL agent, which is a
   *favourable* result and currently goes unreported because nothing counted
   tokens.
