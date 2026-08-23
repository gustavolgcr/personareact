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
