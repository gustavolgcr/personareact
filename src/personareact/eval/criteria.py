"""G-Eval criteria and their rubrics.

Named after what they measure. In the original, the function that Experiment 1
actually called was ``get_criteria_experiment_2()`` while
``get_criteria_experiment_1()`` held a different, unused rubric -- an easy way
to grade a chapter with the wrong instrument.

The rubric text is preserved from the thesis (Appendix B, Códigos-fonte 1-3) so
the corrected runs stay comparable with the published ones. One typo is fixed
("perfol" -> "perfil") and noted in the errata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Judge criteria, keyed by the stable id used in configs and result files.
CRITERIA: dict[str, dict[str, Any]] = {
    "language_and_style": {
        "display_name": "Linguagem e Estilo Apropriados",
        "criteria": (
            "Avalie se o estilo da resposta soa natural para o perfil informado, "
            "com palavras, tom e jeito de falar que combinem com esse tipo de perfil."
        ),
        "rubric": [
            (
                (0, 2),
                "A linguagem está fora do tom esperado, usando palavras difíceis, "
                "jargões ou um estilo que não combina com o perfil.",
            ),
            (
                (3, 5),
                "Há tentativa de adaptar o estilo, mas ainda soa genérico ou forçado em várias partes.",
            ),
            (
                (6, 8),
                "O estilo é geralmente bom e condiz com o perfil, mesmo que tenha "
                "pequenos momentos de desalinhamento.",
            ),
            (
                (9, 10),
                "O estilo é totalmente natural e fluido, combinando perfeitamente com o perfil do início ao fim.",
            ),
        ],
    },
    "data_interpretation": {
        "display_name": "Interpretação e Explicação dos Dados",
        "criteria": (
            "Avalie se a resposta interpreta corretamente os dados apresentados e "
            "fornece explicações úteis e contextualizadas para o perfil informado."
        ),
        "rubric": [
            (
                (0, 2),
                "A resposta apenas menciona dados brutos ou incorretos, sem interpretar ou contextualizar.",
            ),
            (
                (3, 5),
                "Há tentativa de interpretação, mas ela é vaga, superficial ou genérica, sem agregar valor real.",
            ),
            (
                (6, 8),
                "Os dados são bem interpretados, com explicações que ajudam o perfil a compreender sua relevância.",
            ),
            (
                (9, 10),
                "A resposta combina dados específicos, interpretação correta e sugestões "
                "relevantes, oferecendo uma explicação clara e estratégica adaptada ao perfil.",
            ),
        ],
    },
    "style_consistency": {
        "display_name": "Coerência do Estilo ao Longo do Texto",
        "criteria": (
            "Avalie se o estilo adotado é mantido de forma consistente ao longo da "
            "resposta, sem mudanças abruptas ou incoerentes no tom, vocabulário ou estrutura."
        ),
        "rubric": [
            (
                (0, 2),
                "A resposta apresenta variações bruscas no estilo, alternando entre registros diferentes ou incoerentes.",
            ),
            (
                (3, 5),
                "O estilo tenta ser consistente, mas há mudanças perceptíveis de tom ou vocabulário em partes relevantes.",
            ),
            (
                (6, 8),
                "O estilo é predominantemente coerente, com apenas uma ou outra leve oscilação que não compromete o todo.",
            ),
            (
                (9, 10),
                "A resposta mantém o mesmo padrão de estilo do início ao fim, com tom e vocabulário uniformes e adequados.",
            ),
        ],
    },
}


@dataclass(frozen=True)
class CriterionSpec:
    key: str
    display_name: str
    criteria: str
    rubric: list[tuple[tuple[int, int], str]]


def get_criterion(key: str) -> CriterionSpec:
    if key not in CRITERIA:
        raise KeyError(f"unknown criterion {key!r}; available: {', '.join(CRITERIA)}")
    spec = CRITERIA[key]
    return CriterionSpec(
        key=key,
        display_name=spec["display_name"],
        criteria=spec["criteria"],
        rubric=spec["rubric"],
    )


def _register_with_deepeval(
    model: str, *, usd_per_1m_input: float, usd_per_1m_output: float
) -> None:
    """Make a model deepeval has never heard of usable as a judge.

    deepeval 2.9.7 carries three hardcoded registries, and a model missing from
    any of them fails in a different way:

    ``valid_gpt_models``
        ``GPTModel.__init__`` raises ``ValueError: Invalid model`` outright.
    ``structured_outputs_models``
        no error, but G-Eval silently falls back off the structured-output
        path, so the rubric scoring becomes free-text parsing. This is the
        dangerous one -- it degrades quietly.
    ``model_pricing``
        ``calculate_cost`` raises ``KeyError: 'input'`` after the call has
        already been paid for.

    The library predates every model released since. Registering is the only
    way to use a current judge without forking it, and doing it in one place
    keeps the workaround visible.
    """
    try:
        from deepeval.models.llms import openai_model
    except ImportError:  # pragma: no cover - handled by the caller
        return

    for registry in ("valid_gpt_models", "structured_outputs_models", "json_mode_models"):
        models = getattr(openai_model, registry, None)
        if isinstance(models, list) and model not in models:
            models.append(model)

    pricing = getattr(openai_model, "model_pricing", None)
    if isinstance(pricing, dict) and model not in pricing:
        # A dated snapshot costs what its alias costs, so inherit when we can
        # and fall back to the price declared in the judge config.
        base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
        pricing[model] = pricing.get(
            base,
            {
                "input": usd_per_1m_input / 1_000_000,
                "output": usd_per_1m_output / 1_000_000,
            },
        )


def build_judge_model(config: Any) -> Any:
    """Build the deepeval model object for a judge config.

    Returns a configured ``GPTModel`` instance rather than a bare model name,
    because deepeval passes ``temperature`` on every completion call and
    defaults it to 0. Reasoning models reject a non-default temperature, so a
    modern judge has to be handed an instance with the right value rather than
    a string.
    """
    from deepeval.models.llms.openai_model import GPTModel

    _register_with_deepeval(
        config.model,
        usd_per_1m_input=config.usd_per_1m_input,
        usd_per_1m_output=config.usd_per_1m_output,
    )
    return GPTModel(model=config.model, temperature=config.temperature)


def build_geval_metrics(keys: list[str], *, judge: Any) -> list[Any]:
    """Instantiate deepeval G-Eval metrics for a judge config.

    The model is passed explicitly on every metric. The original left
    ``# model=OPENAI_MODEL`` commented out, so deepeval silently used its own
    default (``gpt-4o``) and nothing in the repository recorded which model had
    graded the thesis.
    """
    try:
        from deepeval.metrics import GEval
        from deepeval.metrics.g_eval import Rubric
        from deepeval.test_case import LLMTestCaseParams
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "judging needs the `judge` extra:\n    uv pip install -e '.[judge]'"
        ) from exc

    judge_model = build_judge_model(judge)

    metrics = []
    for key in keys:
        spec = get_criterion(key)
        metrics.append(
            GEval(
                name=spec.display_name,
                criteria=spec.criteria,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.CONTEXT,
                ],
                rubric=[
                    Rubric(score_range=score_range, expected_outcome=outcome)
                    for score_range, outcome in spec.rubric
                ],
                model=judge_model,
                verbose_mode=False,
            )
        )
    return metrics
