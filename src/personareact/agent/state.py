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

    def observations(self) -> list[str]:
        """Just the observations -- the ``O*`` of the thesis' formalisation."""
        prefix = "Observation: "
        return [line[len(prefix) :] for line in self.scratchpad if line.startswith(prefix)]

    def tailor_input(self, sees: str) -> str:
        """What the Tailor step is shown.

        Section 4.2 formalises the personalization function as
        ``tau : P x O* -> M`` -- profile and *observations*. The original
        implementation passed the whole scratchpad, which also contains the
        thoughts, the actions and, critically, the line ``Standard Answer:
        ...``. So the Tailor was handed a finished answer to rewrite rather
        than the retrieved facts to compose from, which weakens section 4.4's
        claim of "dissociação entre o que o agente observa e o que é
        comunicado": the wording of the neutral answer anchors the
        personalized one.

        ``context`` reproduces the thesis behaviour and is the default so the
        comparison holds. ``observations`` implements the formalisation, and
        makes the difference between the two a measurable ablation rather than
        an unnoticed gap.
        """
        if sees == "observations":
            observations = self.observations()
            if not observations:
                # Nothing was retrieved (a tool-free approach, or a failed
                # lookup): fall back rather than personalize against nothing.
                return self.history()
            return "\n".join(f"Observation: {o}" for o in observations)
        return self.history()
