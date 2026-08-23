"""Graph state.

Two changes from the original ``PersonaReActState``:

* ``profile`` is optional. Section 4.4 of the thesis claims PersonaReAct is
  backward compatible with ReAct -- "se nenhum perfil for fornecido, o modelo
  pode ser simplificado para o ReAct tradicional" -- but the original state made
  ``profile`` a required field and always ran ``persona_node``, so the claimed
  property was never actually exposed. Here a ``None`` profile is a valid state
  and the Tailor step is skipped, which also makes the ablation a config flag.

* tool callables are no longer stored in the state. The original put a
  ``Dict[str, Callable]`` inside the LangGraph state, which makes the state
  unserialisable and rules out checkpointing or resuming a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    question: str
    profile: str | None = None
    tool_names: list[str] = field(default_factory=list)

    iteration_count: int = 0
    scratchpad: list[str] = field(default_factory=list)

    action: str = ""
    action_input: str = ""
    observation: str = ""

    #: Answer produced by the Reason/Act cycle, before personalization.
    standard_answer: str = ""
    #: Answer produced by the Tailor step. Empty when the approach has no Tailor.
    personalized_answer: str = ""

    #: Set when the loop hit ``max_iterations`` without the model concluding.
    truncated: bool = False
    #: Every SQL call the Text-to-SQL tool made, for auditing.
    sql_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def final_answer(self) -> str:
        """What the user sees: the personalized answer when there is one."""
        return self.personalized_answer or self.standard_answer

    def history(self) -> str:
        return "\n".join(self.scratchpad)
