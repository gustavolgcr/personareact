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


def _ensure_pricing(model: str) -> None:
    """Teach deepeval the price of a dated model snapshot.

    deepeval 2.9.7 accepts ``gpt-4o-2024-11-20`` as a valid model but prices
    only the floating alias ``gpt-4o``, so pinning the judge -- which is the
    whole point of SV-13 -- crashes its cost calculation with ``KeyError:
    'input'``. Rather than give up the pin, we copy the base alias's price onto
    the snapshot.

    This is a shim around a library limitation, not a correction to a price: a
    dated snapshot and its alias cost the same at the moment the alias points
    at it.
    """
    try:
        from deepeval.models.llms import openai_model
    except ImportError:  # pragma: no cover - handled by the caller
        return

    pricing = getattr(openai_model, "model_pricing", None)
    if not isinstance(pricing, dict) or model in pricing:
        return

    # "gpt-4o-2024-11-20" -> "gpt-4o";  "o3-mini-2025-01-31" -> "o3-mini"
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    if base in pricing:
        pricing[model] = pricing[base]


def build_geval_metrics(keys: list[str], *, model: str) -> list[Any]:
    """Instantiate deepeval G-Eval metrics with the judge model pinned.

    ``model`` is passed explicitly on every metric. The original left
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

    _ensure_pricing(model)

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
                model=model,
                verbose_mode=False,
            )
        )
    return metrics
