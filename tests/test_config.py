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

    def test_only_personareact_and_enriched_use_the_profile(self):
        using = {n for n in list_configs("approach") if ApproachConfig.load(n).uses_profile}
        assert using == {"personareact", "react_enriched"}

    def test_enriched_suffix_carries_the_profile_placeholder(self):
        suffix = ApproachConfig.load("react_enriched").question_suffix
        assert suffix and "{profile}" in suffix

    def test_no_other_approach_alters_the_question(self):
        for name in list_configs("approach"):
            if name == "react_enriched":
                continue
            assert ApproachConfig.load(name).question_suffix is None


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
    def test_every_model_config_loads_and_is_pinned_to_a_snapshot(self):
        for name in list_configs("model"):
            config = ModelConfig.load(name)
            assert any(char.isdigit() for char in config.model.split("-")[-1]), (
                f"{name} must name a dated snapshot, not a floating alias"
            )

    def test_reasoning_models_send_no_sampling_parameters(self):
        for name in list_configs("model"):
            config = ModelConfig.load(name)
            if config.reasoning:
                assert config.sampling_kwargs() == {}

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
    def test_judge_config_loads(self):
        config = JudgeConfig.load("geval-gpt4o-2024-11-20")
        assert config.repetitions == 5
        assert config.criteria

    def test_a_floating_alias_is_rejected(self, tmp_path, monkeypatch):
        """The judge model must be a dated snapshot.

        `gpt-4o` is repointed by OpenAI over time; the original relied on it as
        deepeval's silent default, which makes the published scores
        unrecoverable.
        """
        import personareact.config as config_module

        directory = tmp_path / "judge"
        directory.mkdir(parents=True)
        (directory / "loose.yaml").write_text("model: gpt-4o\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

        with pytest.raises(ConfigError, match="floating alias"):
            JudgeConfig.load("loose")


class TestExperiments:
    def test_e1_expands_to_the_full_sweep(self):
        cells = ExperimentConfig.load("e1").cells()
        assert len(cells) == 2 * 3 * 7 * 6 * 2  # domain x approach x model x profile x difficulty

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
        assert len(config.cells()) == 6
        assert config.limit_questions == 1

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
