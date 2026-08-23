"""Tests for prompt scoping, tool-error detection and graph assembly.

The prompt tests guard SV-06: a single shared prompt file meant the medical
case study ran with football instructions ("A player can have multiple segments
in the same match") and football personas as its tone examples.
"""

from __future__ import annotations

import pytest

from personareact.agent.nodes import AgentRuntime
from personareact.agent.state import AgentState
from personareact.config import ApproachConfig, DomainConfig, ModelConfig
from personareact.prompts import (
    PromptError,
    load_profile_guidance,
    manifest_hashes,
    prompt_hash,
    render,
    render_profile_guidance,
)
from personareact.tools.text_to_sql import TOOL_ERROR_PREFIX, is_tool_error


class TestPromptScoping:
    def test_each_domain_has_its_own_text_to_sql_prompt(self):
        soccer = render("text_to_sql", scope="soccer", dialect="SQLite")
        covid = render("text_to_sql", scope="covid", dialect="SQLite")
        assert soccer != covid

    def test_football_instructions_never_reach_the_medical_prompt(self):
        covid = render("text_to_sql", scope="covid", dialect="SQLite").lower()
        for leak in ("player", "match", "segments in the same match", "sprint"):
            assert leak not in covid

    def test_medical_instructions_never_reach_the_football_prompt(self):
        soccer = render("text_to_sql", scope="soccer", dialect="SQLite").lower()
        for leak in ("id_paciente", "id_atendimento", "de_analito"):
            assert leak not in soccer

    def test_profile_guidance_is_domain_scoped(self):
        soccer = set(load_profile_guidance("soccer"))
        covid = set(load_profile_guidance("covid"))
        assert soccer & covid == set()

    def test_every_domain_profile_has_guidance(self):
        for name in ("soccer", "covid"):
            domain = DomainConfig.load(name)
            guidance = load_profile_guidance(domain.prompt_dir)
            assert set(domain.profiles) == set(guidance)

    def test_tailor_prompt_receives_the_right_domain_examples(self):
        rendered = render(
            "tailor",
            question="q",
            profile="Médico",
            history="h",
            profile_guidance=render_profile_guidance("covid"),
        )
        assert "Enfermeiro" in rendered
        assert "Torcedor do Time" not in rendered

    def test_a_missing_placeholder_is_an_error_not_a_silent_gap(self):
        with pytest.raises(PromptError, match="profile_guidance"):
            render("tailor", question="q", profile="p", history="h")

    def test_prompt_hashes_are_stable_and_recorded(self):
        first = prompt_hash("tailor")
        assert first == prompt_hash("tailor")
        hashes = manifest_hashes("soccer")
        assert "pt_BR/shared/tailor.txt" in hashes
        assert "pt_BR/soccer/text_to_sql.txt" in hashes
        assert "pt_BR/soccer/profiles.yaml" in hashes

    def test_manifest_hashes_do_not_include_the_other_domain(self):
        assert not any("covid" in key for key in manifest_hashes("soccer"))


class TestToolErrors:
    def test_a_tool_error_is_recognised(self):
        assert is_tool_error(f"{TOOL_ERROR_PREFIX}: connection refused")

    def test_a_real_answer_is_not(self):
        assert not is_tool_error("A velocidade média foi de 1.36 m/s.")

    def test_empty_is_not_an_error_string(self):
        assert not is_tool_error("")


class TestGraphAssembly:
    """The graph is built from flags, so the wiring is worth asserting."""

    @staticmethod
    def runtime(approach: str) -> AgentRuntime:
        from personareact.llm import LLMClient

        return AgentRuntime(
            approach=ApproachConfig.load(approach),
            domain=DomainConfig.load("soccer"),
            llm=LLMClient(config=ModelConfig.load("gpt-4.1-mini"), api_key="test"),
            tool=None,
        )

    @pytest.mark.parametrize("approach", ["standard", "cot"])
    def test_tool_free_approaches_compile_without_a_tool(self, approach):
        from personareact.agent.graph import build_graph

        assert build_graph(self.runtime(approach)) is not None

    def test_personareact_graph_contains_the_tailor_node(self):
        from personareact.agent.graph import build_graph

        runtime = self.runtime("personareact")
        graph = build_graph(runtime)
        assert "tailor" in graph.get_graph().nodes

    def test_react_graph_has_no_tailor_node(self):
        from personareact.agent.graph import build_graph

        graph = build_graph(self.runtime("react"))
        assert "tailor" not in graph.get_graph().nodes


class TestTailorBackwardCompatibility:
    """Section 4.4: with no profile, PersonaReAct must degrade to ReAct.

    The original made ``profile`` a required state field and always ran the
    personalization node, so the claimed property did not exist.
    """

    def test_no_profile_means_no_personalization(self):
        from personareact.agent.nodes import make_tailor_node

        runtime = TestGraphAssembly.runtime("personareact")
        node = make_tailor_node(runtime)
        state = AgentState(question="q", profile=None, standard_answer="plain answer")
        assert node(state) == {"personalized_answer": ""}

    def test_final_answer_falls_back_to_the_standard_answer(self):
        state = AgentState(question="q", standard_answer="plain", personalized_answer="")
        assert state.final_answer == "plain"

    def test_personalized_answer_wins_when_present(self):
        state = AgentState(question="q", standard_answer="plain", personalized_answer="tailored")
        assert state.final_answer == "tailored"
