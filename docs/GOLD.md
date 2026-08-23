# Gold answers

The factuality check (`personareact.eval.factuality`) works in two modes.

**Without gold values** — the default — it verifies *grounding*: every number
an answer states must be traceable, within a rounding tolerance, to a number
the Text-to-SQL tool actually retrieved. This catches invented figures and unit
conversions, and it needs no extra authoring.

**With gold values** it also verifies *agreement*: the known correct value must
appear in the answer. This is stronger, and it needs one SQL query per neutral
question — 40 in total (2 domains × 2 difficulties × 10 questions).

## Authoring them

Add `gold_sql` and `gold_answer` to the neutral rows of
`data/questions.jsonl`. The profile variants inherit through `neutral_id`, so
only the 40 neutral questions need writing.

```json
{
  "question_id": "soccer-easy-01-neutral",
  "neutral_id": "soccer-easy-01-neutral",
  "domain": "soccer",
  "difficulty": "easy",
  "profile": "neutral",
  "text": "Qual foi a velocidade média de Bruno Pacheco no jogo contra o Ceará no dia 1º de abril de 2023?",
  "gold_sql": "SELECT AVG(es.Velocidade_Media) FROM Estatisticas_Segmento es JOIN Segmentos s ON s.ID_Segmento = es.ID_Segmento JOIN Jogadores j ON j.ID_Jogador = s.ID_Jogador JOIN Jogos g ON g.ID_Jogo = s.ID_Jogo WHERE j.Nome_Jogador = 'Bruno Pacheco' AND g.Data_Jogo = '2023-04-01'",
  "gold_answer": "1.3622634444889454 m/s"
}
```

Keep the unit in `gold_answer`. Units are exactly where the original answers
went wrong: a legacy answer reports "1.362 km/h" for a column stored in metres
per second (`docs/ERRATA.md`, SV-10).

## Checking your work

```bash
personareact corpus stats     # reports how many questions carry a gold answer
```

Run each `gold_sql` against the database before committing it — the corrected
`View_ResumoPorJogadorEPartida` makes most of these one-liners, but the view
that shipped with the thesis returned figures inflated by a median factor of
43× (SV-15), so do not trust a value copied from an old notebook.
