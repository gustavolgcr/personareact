"""Tests for configuration resolution and the approach matrix."""

from __future__ import annotations

import pytest

from personareact.config import (
    ApproachConfig,
    ConfigError,
    DomainConfig,
    ExperimentConfig,
    JudgeConfig,
    ModelConfig,
    list_configs,
    slugify,
)


class TestApproaches:
    def test_every_approach_config_loads(self):
        for name in list_configs("approach"):
            assert ApproachConfig.load(name).name == name

    @pytest.mark.parametrize(
        ("name", "reasoning", "tools", "tailor"),
        [
            ("standard", False, False, False),
            ("cot", True, False, False),
            ("act", False, True, False),
            ("react", True, True, False),
            ("react_enriched", True, True, False),
            ("personareact", True, True, True),
        ],
    )
    def test_approach_matrix(self, name, reasoning, tools, tailor):
        config = ApproachConfig.load(name)
        assert (config.use_reasoning, config.use_tools, config.use_tailor) == (
            reasoning,
            tools,
            tailor,
        )

    def test_only_the_personalizing_approaches_use_the_profile(self):
        using = {n for n in list_configs("approach") if ApproachConfig.load(n).uses_profile}
        assert using == {
            "personareact",
            "personareact-thesis",
            "react_enriched",
            "react_enriched-thesis",
        }

    def test_the_profile_blind_approaches_stay_blind(self):
        for name in ("standard", "cot", "act", "react", "react-thesis"):
            assert ApproachConfig.load(name).uses_profile is False

    def test_tool_using_approaches_declare_a_known_toolset(self):
        for name in list_configs("approach"):
            config = ApproachConfig.load(name)
            assert config.toolset in {"primitives", "agent"}

    def test_the_defaults_follow_chapter_four(self):
        """The primary configs implement what the thesis *describes*."""
        personareact = ApproachConfig.load("personareact")
        assert personareact.toolset == "primitives"  # the loop really interleaves
        assert personareact.tailor_sees == "observations"  # tau(p, O*)

    def test_the_thesis_variants_reproduce_what_was_run(self):
        """The `-thesis` configs implement what the thesis *did*."""
        for name in ("react-thesis", "react_enriched-thesis", "personareact-thesis"):
            config = ApproachConfig.load(name)
            assert config.toolset == "agent"
            assert config.max_iterations == 5
        assert ApproachConfig.load("personareact-thesis").tailor_sees == "context"

    def test_act_only_keeps_the_nested_agent(self):
        """With primitives there is no single call that answers a question."""
        assert ApproachConfig.load("act").toolset == "agent"

    def test_tailor_sees_is_a_known_mode(self):
        for name in list_configs("approach"):
            assert ApproachConfig.load(name).tailor_sees in {"context", "observations"}

    def test_enriched_suffix_carries_the_profile_placeholder(self):
        suffix = ApproachConfig.load("react_enriched").question_suffix
        assert suffix and "{profile}" in suffix

    def test_no_other_approach_alters_the_question(self):
        enriched = {"react_enriched", "react_enriched-thesis"}
        for name in list_configs("approach"):
            if name in enriched:
                continue
            assert ApproachConfig.load(name).question_suffix is None

    def test_both_enriched_variants_use_the_same_suffix(self):
        """The one-line suffix from react/src/main.py:86, in both configs."""
        assert (
            ApproachConfig.load("react_enriched").question_suffix
            == ApproachConfig.load("react_enriched-thesis").question_suffix
        )


class TestDomains:
    def test_every_domain_config_loads(self):
        for name in list_configs("domain"):
            assert len(DomainConfig.load(name).profiles) == 6

    def test_domains_have_distinct_prompt_directories(self):
        directories = {DomainConfig.load(n).prompt_dir for n in list_configs("domain")}
        assert len(directories) == len(list_configs("domain"))

    def test_soccer_declares_the_summary_views(self):
        assert DomainConfig.load("soccer").expects_views == [
            "View_EstatisticasComJogadorJogo",
            "View_ResumoPorJogadorEPartida",
        ]


class TestModels:
    def test_every_model_names_a_dated_snapshot_or_admits_it_cannot(self):
        """A model id must be reproducible, or say out loud that it is not.

        OpenAI stopped publishing dated snapshots with the GPT-5.6 family:
        ``gpt-5.6-luna`` is itself the only snapshot name. That is a real
        reproducibility regression, so the config has to declare it rather
        than let a floating id slip through unnoticed.
        """
        for name in list_configs("model"):
            config = ModelConfig.load(name)
            dated = any(char.isdigit() for char in config.model.split("-")[-1])
            assert dated or config.no_dated_snapshot, (
                f"{name} names neither a dated snapshot nor declares no_dated_snapshot"
            )

    def test_the_thesis_models_are_all_dated(self):
        """Whatever else changes, the published runs stay identifiable."""
        for name in ("o3-mini", "o4-mini", "gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini"):
            config = ModelConfig.load(name)
            assert not config.no_dated_snapshot
            assert any(char.isdigit() for char in config.model.split("-")[-1])

    def test_reasoning_models_send_no_sampling_parameters(self):
        """o-series and GPT-5 reject temperature, top_p and friends.

        ``reasoning_effort`` is not a sampling parameter -- it is the only knob
        those models do accept, and in the GPT-5.6 generation it is the axis H4
        is actually about.
        """
        for name in list_configs("model"):
            config = ModelConfig.load(name)
            if config.reasoning:
                assert set(config.sampling_kwargs()) <= {"reasoning_effort"}

    def test_the_reasoning_dial_is_a_known_level(self):
        levels = {"none", "low", "medium", "high", "xhigh", "max"}
        for name in list_configs("model"):
            effort = ModelConfig.load(name).reasoning_effort
            assert effort is None or effort in levels

    def test_only_reasoning_models_set_an_effort(self):
        for name in list_configs("model"):
            config = ModelConfig.load(name)
            if config.reasoning_effort is not None:
                assert config.reasoning, f"{name} sets an effort but is not a reasoning model"

    def test_non_reasoning_models_send_them_explicitly(self):
        config = ModelConfig.load("gpt-4.1-mini")
        kwargs = config.sampling_kwargs()
        assert kwargs["temperature"] == 1.0
        assert kwargs["top_p"] == 1.0
        assert kwargs["frequency_penalty"] == 0.0
        assert kwargs["seed"] is not None

    def test_sampling_parameters_are_not_smuggled_through_model_kwargs(self):
        assert "model_kwargs" not in ModelConfig.load("gpt-4.1-mini").sampling_kwargs()

    def test_sql_model_defaults_to_the_agent_model(self):
        config = ModelConfig.load("o3-mini")
        assert config.effective_sql_model == config.model


class TestJudge:
    def test_the_default_judge_is_the_current_flagship(self):
        config = JudgeConfig.load("geval-gpt-5.6-sol")
        assert config.model == "gpt-5.6-sol"
        assert config.repetitions == 5
        assert config.criteria

    def test_the_legacy_judge_is_kept_for_comparability(self):
        """Every published score came from this model; it must stay loadable."""
        config = JudgeConfig.load("geval-gpt4o-2024-11-20")
        assert config.model == "gpt-4o-2024-11-20"

    def test_a_reasoning_judge_uses_the_only_temperature_it_accepts(self):
        """deepeval sends temperature on every call and defaults it to 0.

        Reasoning models reject anything but the default, so the config has to
        carry 1.0 or the judge fails on its first request.
        """
        assert JudgeConfig.load("geval-gpt-5.6-sol").temperature == 1.0

    def test_the_legacy_judge_keeps_its_original_temperature(self):
        assert JudgeConfig.load("geval-gpt4o-2024-11-20").temperature == 0.0

    def test_a_judge_without_a_dated_snapshot_must_say_so(self):
        assert JudgeConfig.load("geval-gpt-5.6-sol").no_dated_snapshot is True
        assert JudgeConfig.load("geval-gpt4o-2024-11-20").no_dated_snapshot is False

    def test_every_judge_declares_its_price(self):
        """deepeval prices only models it shipped with; the rest must declare."""
        for name in list_configs("judge"):
            config = JudgeConfig.load(name)
            assert config.usd_per_1m_input > 0
            assert config.usd_per_1m_output > 0

    @pytest.mark.parametrize("alias", ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"])
    def test_a_repointed_alias_is_rejected(self, tmp_path, monkeypatch, alias):
        """These are aliases OpenAI moves; naming one loses the scores."""
        import personareact.config as config_module

        directory = tmp_path / "judge"
        directory.mkdir(parents=True)
        (directory / "loose.yaml").write_text(f"model: {alias}\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

        with pytest.raises(ConfigError, match="repoints"):
            JudgeConfig.load("loose")

    def test_an_undated_model_must_declare_it_has_no_snapshot(self, tmp_path, monkeypatch):
        import personareact.config as config_module

        directory = tmp_path / "judge"
        directory.mkdir(parents=True)
        (directory / "undated.yaml").write_text("model: some-new-model\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

        with pytest.raises(ConfigError, match="no_dated_snapshot"):
            JudgeConfig.load("undated")

        (directory / "undated.yaml").write_text(
            "model: some-new-model\nno_dated_snapshot: true\n", encoding="utf-8"
        )
        assert JudgeConfig.load("undated").model == "some-new-model"


class TestExperiments:
    def test_e1_expands_to_the_full_sweep(self):
        cells = ExperimentConfig.load("e1").cells()
        assert len(cells) == 2 * 3 * 7 * 6 * 2  # domain x approach x model x profile x difficulty

    def test_domains_declare_units_for_their_measures(self):
        """Units travel with the observation, so they cannot be lost downstream."""
        soccer = DomainConfig.load("soccer")
        assert soccer.units["Velocidade_Media"] == "m/s"
        assert soccer.units["Tempo_Parado"] == "s"

    def test_every_approach_runs_on_every_profile(self):
        """The fix for SV-03.

        The original ran plain ReAct once per (difficulty, model) against the
        neutral questions and reused that answer set for all six profiles.
        """
        cells = ExperimentConfig.load("e1").cells()
        for approach in ("react", "react_enriched", "personareact"):
            profiles = {c.profile for c in cells if c.approach.name == approach}
            assert len(profiles) == 12  # 6 per domain, two domains

    def test_smoke_is_cheap(self):
        config = ExperimentConfig.load("smoke")
        assert len(config.cells()) == len(config.approaches)
        assert config.limit_questions == 1

    def test_smoke_covers_both_the_corrected_and_reproduced_paths(self):
        approaches = set(ExperimentConfig.load("smoke").approaches)
        assert {"personareact", "personareact-thesis"} <= approaches

    def test_e1_and_e1_thesis_are_the_same_shape(self):
        """The pair differs only in how each approach is implemented."""
        corrected = ExperimentConfig.load("e1")
        thesis = ExperimentConfig.load("e1-thesis")
        assert len(corrected.cells()) == len(thesis.cells())
        assert [f"{a}-thesis" for a in corrected.approaches] == thesis.approaches

    def test_cell_slugs_are_unique(self):
        cells = ExperimentConfig.load("e1").cells()
        assert len({c.slug for c in cells}) == len(cells)


class TestSlugify:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Técnico do Time", "tecnico_do_time"),
            ("Pesquisador em Saúde Pública", "pesquisador_em_saude_publica"),
            ("Familiar de um Paciente", "familiar_de_um_paciente"),
        ],
    )
    def test_matches_the_legacy_filenames(self, text, expected):
        """Kept byte-compatible so legacy artefacts still resolve."""
        assert slugify(text) == expected
