"""Graph assembly and single-question execution.

One graph builder covers all six approaches. Which nodes exist and how they are
wired is derived from the approach config, so ``standard``, ``cot``, ``act``,
``react``, ``react_enriched`` and ``personareact`` are six YAML files rather
than six copy-pasted projects with six diverged copies of the same bug.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from ..config import ApproachConfig, DomainConfig, ModelConfig
from ..corpus.schema import Question
from ..llm import LLMClient, LLMError, Usage, build_client
from ..tools.text_to_sql import TextToSQLTool, is_tool_error
from .nodes import (
    AgentRuntime,
    make_act_node,
    make_act_only_node,
    make_direct_node,
    make_reason_node,
    make_router,
    make_tailor_node,
)
from .state import AgentState


def build_graph(rt: AgentRuntime):
    """Compile the LangGraph state machine for one approach."""
    builder = StateGraph(AgentState)
    approach = rt.approach
    terminal = "tailor" if approach.use_tailor else END

    if approach.use_tailor:
        builder.add_node("tailor", make_tailor_node(rt))
        builder.add_edge("tailor", END)

    if not approach.use_tools:
        # standard / cot: a single model call.
        builder.add_node("reason", make_direct_node(rt))
        builder.set_entry_point("reason")
        builder.add_edge("reason", terminal)
        return builder.compile()

    if not approach.use_reasoning:
        # act: a single tool call, no reasoning trace.
        builder.add_node("act_only", make_act_only_node(rt))
        builder.set_entry_point("act_only")
        builder.add_edge("act_only", terminal)
        return builder.compile()

    # react / react_enriched / personareact: the full Reason-Act cycle.
    builder.add_node("reason", make_reason_node(rt))
    builder.add_node("act", make_act_node(rt))
    builder.set_entry_point("reason")
    builder.add_conditional_edges(
        "reason",
        make_router(rt, terminal="tailor" if approach.use_tailor else "__end__"),
        {
            "act": "act",
            "reason": "reason",
            "tailor": "tailor" if approach.use_tailor else END,
            "__end__": END,
        },
    )
    builder.add_edge("act", "reason")
    return builder.compile()


@dataclass
class AgentResult:
    answer: str
    standard_answer: str
    personalized_answer: str
    scratchpad: list[str]
    iterations: int
    truncated: bool
    sql_calls: list[dict[str, Any]]
    ok: bool
    error: str | None
    elapsed_s: float


class Agent:
    """A configured agent, reusable across the questions of one cell."""

    def __init__(
        self,
        *,
        approach: ApproachConfig,
        domain: DomainConfig,
        model: ModelConfig,
        api_key: str | None = None,
        verbose: bool = False,
    ):
        self.approach = approach
        self.domain = domain
        self.model = model
        self.llm: LLMClient = build_client(model, api_key=api_key)
        tool = (
            TextToSQLTool(domain=domain, model=model, api_key=api_key, verbose=verbose)
            if approach.use_tools
            else None
        )
        self.runtime = AgentRuntime(approach=approach, domain=domain, llm=self.llm, tool=tool)
        self.graph = build_graph(self.runtime)

    @property
    def usage(self) -> Usage:
        """Everything this agent spent, outer loop and nested SQL agent together.

        The nested Text-to-SQL agent runs its own model and is the larger
        consumer -- it lists tables, reads schemas, then queries. Reporting only
        the outer loop understated a run's cost by roughly five times, and made
        the ``act`` approach look free.
        """
        total = Usage()
        total.add(self.llm.usage)
        if self.runtime.tool is not None:
            total.add(self.runtime.tool.usage)
        return total

    def question_text(self, question: Question, profile: str | None) -> str:
        """Apply the approach's question transformation.

        ``react_enriched`` is the only approach that alters the question: it
        appends the profile as a suffix. In the original code this was a
        commented-out line in ``react/src/main.py`` that also required
        hand-editing the output path, so nothing recorded which of the two
        conditions any given file belonged to.
        """
        if self.approach.question_suffix and profile:
            return question.text + self.approach.question_suffix.format(profile=profile)
        return question.text

    def answer(self, question: Question, profile: str | None = None) -> AgentResult:
        """Run one question.

        ``profile`` is only passed into the state when the approach actually
        uses it. The profile-blind approaches still receive *the profile's*
        phrasing of the question, so every condition answers the same thing --
        the original ran plain ReAct against the neutral questions instead and
        then told the judge it had answered the adapted one.
        """
        if self.runtime.tool is not None:
            self.runtime.tool.calls = []

        state = AgentState(
            question=self.question_text(question, profile),
            profile=profile if self.approach.uses_profile else None,
            tool_names=self.runtime.tool_names,
        )
        state.scratchpad.append(f"Question: {state.question}")

        started = time.perf_counter()
        try:
            raw = self.graph.invoke(state, {"recursion_limit": 64})
        except LLMError as exc:
            return self._failure(str(exc), started)
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            return self._failure(f"{type(exc).__name__}: {exc}", started)

        final = _as_state(raw)
        answer = final.final_answer.strip()
        truncated = not final.standard_answer.strip() and (
            final.iteration_count >= self.approach.max_iterations
        )
        # SV-09: an empty answer is a failure, not a datum. So is a tool error
        # that reached the answer field -- which is exactly how the `act`
        # approach can fail, since its answer *is* the observation. The original
        # wrote empty strings and `[ERRO: 'thought']` into the answer field and
        # the evaluator scored them.
        ok = bool(answer) and not is_tool_error(answer)
        error: str | None = None
        if not answer:
            error = "empty answer"
        elif is_tool_error(answer):
            error = f"tool failure surfaced as the answer: {answer[:120]}"
        if truncated and not answer:
            error = f"loop truncated at {self.approach.max_iterations} iterations with no answer"

        return AgentResult(
            answer=answer,
            standard_answer=final.standard_answer.strip(),
            personalized_answer=final.personalized_answer.strip(),
            scratchpad=list(final.scratchpad),
            iterations=final.iteration_count,
            truncated=truncated,
            sql_calls=list(final.sql_calls),
            ok=ok,
            error=error,
            elapsed_s=round(time.perf_counter() - started, 3),
        )

    def _failure(self, message: str, started: float) -> AgentResult:
        return AgentResult(
            answer="",
            standard_answer="",
            personalized_answer="",
            scratchpad=[],
            iterations=0,
            truncated=False,
            sql_calls=list(self.runtime.tool.calls) if self.runtime.tool else [],
            ok=False,
            error=message,
            elapsed_s=round(time.perf_counter() - started, 3),
        )


def _as_state(raw: Any) -> AgentState:
    """LangGraph returns a dict for dataclass states; normalise it back."""
    if isinstance(raw, AgentState):
        return raw
    known = set(AgentState.__dataclass_fields__)
    return AgentState(**{k: v for k, v in dict(raw).items() if k in known})
