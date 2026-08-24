"""The deepeval registration shim, exercised without calling the API.

deepeval 2.9.7 predates every model released since, and carries three hardcoded
registries. A model missing from one of them fails differently, and the
``structured_outputs_models`` case fails *quietly* -- G-Eval drops off the
structured-output path and the rubric scoring becomes free-text parsing.
"""

from __future__ import annotations

import pytest

from personareact.config import JudgeConfig

deepeval = pytest.importorskip("deepeval", reason="needs the judge extra")


@pytest.fixture
def registries():
    from deepeval.models.llms import openai_model

    saved = {
        name: list(getattr(openai_model, name))
        for name in ("valid_gpt_models", "structured_outputs_models", "json_mode_models")
    }
    saved_pricing = dict(openai_model.model_pricing)
    yield openai_model
    for name, value in saved.items():
        getattr(openai_model, name)[:] = value
    openai_model.model_pricing.clear()
    openai_model.model_pricing.update(saved_pricing)


class TestRegistration:
    def test_the_flagship_is_unknown_to_deepeval_out_of_the_box(self, registries):
        """The premise of the shim. If this ever fails, drop the shim."""
        assert "gpt-5.6-sol" not in registries.valid_gpt_models

    def test_registering_adds_it_to_every_registry(self, registries):
        from personareact.eval.criteria import _register_with_deepeval

        _register_with_deepeval("gpt-5.6-sol", usd_per_1m_input=4.0, usd_per_1m_output=20.0)

        assert "gpt-5.6-sol" in registries.valid_gpt_models
        # The quiet one: without this, G-Eval silently stops using schemas.
        assert "gpt-5.6-sol" in registries.structured_outputs_models
        assert "gpt-5.6-sol" in registries.model_pricing

    def test_the_declared_price_is_converted_per_token(self, registries):
        from personareact.eval.criteria import _register_with_deepeval

        _register_with_deepeval("gpt-5.6-sol", usd_per_1m_input=4.0, usd_per_1m_output=20.0)
        pricing = registries.model_pricing["gpt-5.6-sol"]
        assert pricing["input"] == pytest.approx(4.0 / 1_000_000)
        assert pricing["output"] == pytest.approx(20.0 / 1_000_000)

    def test_a_dated_snapshot_inherits_its_alias_price(self, registries):
        """gpt-4o-2024-11-20 costs what gpt-4o costs."""
        from personareact.eval.criteria import _register_with_deepeval

        _register_with_deepeval("gpt-4o-2024-11-20", usd_per_1m_input=0.0, usd_per_1m_output=0.0)
        assert registries.model_pricing["gpt-4o-2024-11-20"] == registries.model_pricing["gpt-4o"]

    def test_registering_twice_is_harmless(self, registries):
        from personareact.eval.criteria import _register_with_deepeval

        for _ in range(3):
            _register_with_deepeval("gpt-5.6-sol", usd_per_1m_input=4.0, usd_per_1m_output=20.0)
        assert registries.valid_gpt_models.count("gpt-5.6-sol") == 1


@pytest.fixture
def offline_key(monkeypatch):
    """GPTModel builds an OpenAI client eagerly, which wants a key present.

    No request is made -- constructing a client is not a call -- so a dummy
    value keeps these tests free and offline.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


class TestJudgeModelConstruction:
    def test_the_flagship_judge_builds_with_the_right_temperature(self, registries, offline_key):
        """Constructing the model makes no network call, so this is safe."""
        from personareact.eval.criteria import build_judge_model

        model = build_judge_model(JudgeConfig.load("geval-gpt-5.6-sol"))
        assert model.model_name == "gpt-5.6-sol"
        assert model.temperature == 1.0

    def test_the_legacy_judge_still_builds(self, registries, offline_key):
        from personareact.eval.criteria import build_judge_model

        model = build_judge_model(JudgeConfig.load("geval-gpt4o-2024-11-20"))
        assert model.model_name == "gpt-4o-2024-11-20"
        assert model.temperature == 0.0

    def test_metrics_carry_a_configured_instance_not_a_bare_name(self, registries, offline_key):
        """A bare name would let deepeval apply its own temperature default of 0.

        ``metric.evaluation_model`` is only the display name; ``metric.model``
        is the object that makes the calls, and it is the one that has to carry
        the temperature.
        """
        from deepeval.models.llms.openai_model import GPTModel

        from personareact.eval.criteria import build_geval_metrics

        config = JudgeConfig.load("geval-gpt-5.6-sol")
        metrics = build_geval_metrics(config.criteria, judge=config)
        assert len(metrics) == len(config.criteria)
        for metric in metrics:
            assert isinstance(metric.model, GPTModel)
            assert metric.model.model_name == "gpt-5.6-sol"
            assert metric.model.temperature == 1.0

    def test_the_legacy_judge_keeps_temperature_zero_through_the_metric(
        self, registries, offline_key
    ):
        from personareact.eval.criteria import build_geval_metrics

        config = JudgeConfig.load("geval-gpt4o-2024-11-20")
        for metric in build_geval_metrics(config.criteria, judge=config):
            assert metric.model.temperature == 0.0
