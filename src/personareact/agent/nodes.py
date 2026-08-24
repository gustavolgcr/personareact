"""The graph nodes: Reason, Act, Tailor, and the router between them.

The runtime (model client, tool, prompt scope) is captured in a
:class:`AgentRuntime` and closed over by the node functions rather than being
stored in the graph state. The original implementation put the tool callables
into the state itself, which makes the state unserialisable and rules out
LangGraph checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from typing_extensions import TypedDict

from ..config import ApproachConfig, DomainConfig
from ..llm import LLMClient
from ..prompts import render, render_profile_guidance
from ..tools.sql_primitives import TOOL_DESCRIPTIONS, SQLPrimitives
from ..tools.text_to_sql import TextToSQLTool
from .state import AgentState


class ReasoningOutput(TypedDict, total=False):
    thought: Annotated[str, ..., "O raciocínio deste passo"]
    action: Annotated[str, ..., "Nome da ferramenta a chamar"]
    action_input: Annotated[str, ..., "Input em linguagem natural para a ferramenta"]
    standard_answer: Annotated[str, ..., "Resposta não personalizada, em linguagem natural"]


class DirectOutput(TypedDict, total=False):
    thought: Annotated[str, ..., "O raciocínio passo a passo"]
    standard_answer: Annotated[str, ..., "Resposta final, em linguagem natural"]


class TailorOutput(TypedDict):
    personalized_answer: Annotated[str, ..., "Resposta final adaptada ao perfil"]


@dataclass
class AgentRuntime:
    """Everything a node needs, closed over rather than stored in the state."""

    approach: ApproachConfig
    domain: DomainConfig
    llm: LLMClient
    #: The nested Text-to-SQL agent, for `toolset: agent` (the thesis path).
    tool: TextToSQLTool | None = None
    #: The primitive SQL tools, for `toolset: primitives` (the default).
    primitives: SQLPrimitives | None = None

    @property
    def tool_names(self) -> list[str]:
        if self.primitives is not None:
            return list(self.primitives.as_registry())
        return [self.tool.name] if self.tool else []

    @property
    def has_tools(self) -> bool:
        return self.primitives is not None or self.tool is not None

    def describe_tools(self) -> str:
        if self.primitives is not None:
            return "\n".join(
                f"- {name}: {TOOL_DESCRIPTIONS[name]}" for name in self.primitives.as_registry()
            )
        if self.tool is not None:
            return f"- {self.tool.name}: {self.tool.description}"
        return "(nenhuma)"

    def call_tool(self, name: str, argument: str) -> str:
        """Run one action. Unknown names come back as an observation, not a crash."""
        if self.primitives is not None:
            return self.primitives(name, argument)
        if self.tool is not None:
            if name and name != self.tool.name:
                return (
                    f"[Ferramenta desconhecida: {name!r}. "
                    f"Disponíveis: {', '.join(self.tool_names)}]"
                )
            return self.tool(argument)
        raise RuntimeError("call_tool reached with no tools configured")

    def tool_calls(self) -> list[dict[str, Any]]:
        if self.primitives is not None:
            return list(self.primitives.calls)
        return list(self.tool.calls) if self.tool else []

    def reset_calls(self) -> None:
        if self.primitives is not None:
            self.primitives.calls = []
        if self.tool is not None:
            self.tool.calls = []


# --------------------------------------------------------------------------- #
# Reason
# --------------------------------------------------------------------------- #


def make_reason_node(rt: AgentRuntime):
    """Reason step for tool-using approaches (react, react_enriched, personareact)."""

    template = "reason_tools" if rt.primitives is not None else "reason"

    def reason_node(state: AgentState) -> dict[str, Any]:
        values: dict[str, Any] = {"question": state.question, "history": state.history()}
        if template == "reason_tools":
            values["tools"] = rt.describe_tools()
        else:
            values["tool_names"] = ", ".join(rt.tool_names)
        prompt = render(template, **values)
        result = rt.llm.structured(ReasoningOutput, prompt)

        scratchpad = list(state.scratchpad)
        thought = (result.get("thought") or "").strip()
        if thought:
            scratchpad.append(f"Thought: {thought}")

        updates: dict[str, Any] = {
            "scratchpad": scratchpad,
            "action": "",
            "action_input": "",
            "standard_answer": "",
            "iteration_count": state.iteration_count + 1,
        }

        answer = (result.get("standard_answer") or "").strip()
        action = (result.get("action") or "").strip()

        # Truthiness, not key presence. The original tested `if "standard_answer"
        # in result`, which treats a key present-but-empty as a final answer and
        # ends the loop with nothing to say.
        if answer:
            updates["standard_answer"] = answer
            scratchpad.append(f"Standard Answer: {answer}")
        elif action:
            updates["action"] = action
            updates["action_input"] = (result.get("action_input") or state.question).strip()
            scratchpad.append(f"Action: {action}")
            scratchpad.append(f"Action Input: {updates['action_input']}")
        else:
            # Neither an answer nor an action: let the router decide (it will
            # loop back here, bounded by max_iterations).
            scratchpad.append("Thought: (nenhuma ação ou resposta produzida)")

        return updates

    return reason_node


def make_direct_node(rt: AgentRuntime):
    """Single-call step for the tool-free approaches (standard, cot)."""

    template = "cot" if rt.approach.use_reasoning else "standard"

    def direct_node(state: AgentState) -> dict[str, Any]:
        prompt = render(template, question=state.question)
        result = rt.llm.structured(DirectOutput, prompt)

        scratchpad = list(state.scratchpad)
        thought = (result.get("thought") or "").strip()
        if thought:
            scratchpad.append(f"Thought: {thought}")
        answer = (result.get("standard_answer") or "").strip()
        if answer:
            scratchpad.append(f"Standard Answer: {answer}")

        return {
            "scratchpad": scratchpad,
            "standard_answer": answer,
            "iteration_count": state.iteration_count + 1,
        }

    return direct_node


# --------------------------------------------------------------------------- #
# Act
# --------------------------------------------------------------------------- #


def make_act_node(rt: AgentRuntime):
    """Execute the tool with the input the Reason step actually chose."""

    def act_node(state: AgentState) -> dict[str, Any]:
        if not rt.has_tools:
            raise RuntimeError("act_node reached with no tools configured")

        # SV-07: the input is the one the Reason step chose, not the raw
        # question. With primitives that matters on every single step -- a
        # table name, a schema request, a SQL query.
        #
        # An empty argument is *legitimate* for a primitive (`listar_tabelas`
        # takes none), so it is passed through as-is. Only the single-tool
        # path falls back to the question, because there the tool's argument
        # is the question.
        argument = state.action_input
        if not argument and rt.primitives is None:
            argument = state.question
        observation = rt.call_tool(state.action, argument)

        scratchpad = list(state.scratchpad)
        scratchpad.append(f"Observation: {observation}")

        return {
            "scratchpad": scratchpad,
            "observation": observation,
            "action": "",
            "action_input": "",
            "sql_calls": rt.tool_calls(),
        }

    return act_node


def make_act_only_node(rt: AgentRuntime):
    """The Act-only baseline: one tool call, and the observation is the answer."""

    def act_only_node(state: AgentState) -> dict[str, Any]:
        if rt.tool is None:
            raise RuntimeError(
                "the Act-only baseline needs `toolset: agent` -- with primitives "
                "there is no single call that can answer a question"
            )
        observation = rt.tool(state.question)
        scratchpad = list(state.scratchpad)
        scratchpad.append(f"Observation: {observation}")
        scratchpad.append(f"Standard Answer: {observation}")
        return {
            "scratchpad": scratchpad,
            "observation": observation,
            "standard_answer": observation,
            "iteration_count": state.iteration_count + 1,
            "sql_calls": rt.tool_calls(),
        }

    return act_only_node


# --------------------------------------------------------------------------- #
# Tailor
# --------------------------------------------------------------------------- #


def make_tailor_node(rt: AgentRuntime):
    """The personalization function tau(p, O*) from section 4.2 of the thesis."""

    def tailor_node(state: AgentState) -> dict[str, Any]:
        if not state.profile:
            # Section 4.4's backward-compatibility claim, actually implemented:
            # no profile means no personalization, and the agent degrades to
            # plain ReAct instead of personalizing against an empty string.
            return {"personalized_answer": ""}

        prompt = render(
            "tailor",
            question=state.question,
            profile=state.profile,
            history=state.tailor_input(rt.approach.tailor_sees),
            profile_guidance=render_profile_guidance(rt.domain.prompt_dir),
        )
        result = rt.llm.structured(TailorOutput, prompt)
        answer = (result.get("personalized_answer") or "").strip()

        scratchpad = list(state.scratchpad)
        if answer:
            scratchpad.append(f"Personalized: {answer}")

        return {"scratchpad": scratchpad, "personalized_answer": answer}

    return tailor_node


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


def make_router(rt: AgentRuntime, *, terminal: str):
    """Decide the next node after a Reason step.

    ``terminal`` is ``"tailor"`` when the approach has a Tailor step and
    ``"__end__"`` otherwise, so the same router serves react and personareact.
    """

    max_iterations = rt.approach.max_iterations

    def route(state: AgentState) -> str:
        if state.standard_answer:
            return terminal
        if state.iteration_count >= max_iterations:
            return terminal
        if state.action:
            return "act"
        return "reason"

    return route
