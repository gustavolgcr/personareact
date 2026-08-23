"""Build the football tracking database.

Schema and semantics are preserved from the original ``create_database.py``,
with two changes.

**1. The summary view is fixed.** The original
``View_ResumoPorJogadorEPartida`` joined ``Pontos_Trajetoria`` to
``Estatisticas_Segmento`` before aggregating, which fans every segment out to
one row per trajectory point. Segment lengths vary from 3 to 30+ points, so:

* the ``SUM(...)`` columns were multiplied by the point count -- a median
  inflation factor of **43x**, ranging from 3x to 107x across the 727
  player-match rows;
* the ``AVG(...)`` columns became point-weighted rather than segment-weighted
  and were wrong too (Brayan vs Ceará on 2023-02-07: the view reports a mean
  speed of 1.463 m/s against a true 1.026 m/s, and 2,220 sprints against a
  true 62).

This matters beyond data hygiene. That view existed only in the database build
the ReAct baselines queried, not in the one PersonaReAct queried, and the
Text-to-SQL prompt steers the model toward it -- so the baselines could answer
from numbers wrong by up to two orders of magnitude while the treatment
computed its own. See ``docs/ERRATA.md`` (SV-14, SV-15).

The corrected view aggregates the segment statistics directly and pulls total
distance from a scalar subquery.

**2. Bulk inserts.** The original inserted 2.5M trajectory points one
``cursor.execute`` at a time. Same rows, ``executemany``, minutes instead of
hours.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS Jogos (
    ID_Jogo INTEGER PRIMARY KEY AUTOINCREMENT,
    Data_Jogo DATE,
    Adversario VARCHAR(50),
    UNIQUE(Data_Jogo, Adversario)
);

CREATE TABLE IF NOT EXISTS Jogadores (
    ID_Jogador INTEGER PRIMARY KEY AUTOINCREMENT,
    Nome_Jogador VARCHAR(100) UNIQUE,
    Funcao_Jogador VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS Segmentos (
    ID_Segmento INTEGER PRIMARY KEY AUTOINCREMENT,
    ID_Jogo INTEGER,
    ID_Jogador INTEGER,
    ID_Segmento_Original VARCHAR(50),
    Movimento VARCHAR(50),
    FOREIGN KEY(ID_Jogo) REFERENCES Jogos(ID_Jogo),
    FOREIGN KEY(ID_Jogador) REFERENCES Jogadores(ID_Jogador)
);

CREATE TABLE IF NOT EXISTS Pontos_Trajetoria (
    ID_Ponto INTEGER PRIMARY KEY AUTOINCREMENT,
    ID_Segmento INTEGER,
    Indice_Ponto INTEGER,
    Latitude_UTM FLOAT,
    Longitude_UTM FLOAT,
    Latitude_WGS84 FLOAT,
    Longitude_WGS84 FLOAT,
    Elevacao FLOAT,
    Timestamp DATETIME,
    Intervalo_Tempo FLOAT,
    Velocidade FLOAT,
    Direcao FLOAT,
    Distancia FLOAT,
    Zona_UTM INTEGER,
    FOREIGN KEY(ID_Segmento) REFERENCES Segmentos(ID_Segmento)
);

CREATE TABLE IF NOT EXISTS Estatisticas_Segmento (
    ID_Estatistica INTEGER PRIMARY KEY AUTOINCREMENT,
    ID_Segmento INTEGER,
    Velocidade_Media FLOAT,
    Desvio_Velocidade FLOAT,
    Aceleracao_Media FLOAT,
    Desvio_Aceleracao FLOAT,
    Quantidade_Sprints INTEGER,
    Media_Mudanca_Direcao FLOAT,
    Quantidade_Mudanca_Direcao INTEGER,
    Tempo_Parado INTEGER,
    Tempo_Caminhando INTEGER,
    Tempo_Correndo INTEGER,
    Tempo_Sprintando INTEGER,
    Coeficiente_Variacao_Velocidade FLOAT,
    Quantidade_Alta_Aceleracao INTEGER,
    Quantidade_Alta_Desaceleracao INTEGER,
    Media_Curvatura FLOAT,
    Desvio_Curvatura FLOAT,
    FOREIGN KEY(ID_Segmento) REFERENCES Segmentos(ID_Segmento)
);

CREATE INDEX IF NOT EXISTS idx_segmentos_jogo ON Segmentos(ID_Jogo);
CREATE INDEX IF NOT EXISTS idx_segmentos_jogador ON Segmentos(ID_Jogador);
CREATE INDEX IF NOT EXISTS idx_pontos_segmento ON Pontos_Trajetoria(ID_Segmento);
CREATE INDEX IF NOT EXISTS idx_estat_segmento ON Estatisticas_Segmento(ID_Segmento);
"""

#: Views are metadata, so they can be rebuilt on an existing database without
#: touching a single row of data.
VIEWS = """
DROP VIEW IF EXISTS View_EstatisticasComJogadorJogo;
CREATE VIEW View_EstatisticasComJogadorJogo AS
SELECT
    e.ID_Segmento,
    j.Nome_Jogador,
    j.Funcao_Jogador,
    g.Data_Jogo,
    g.Adversario,
    e.Velocidade_Media,
    e.Aceleracao_Media,
    e.Quantidade_Sprints,
    e.Tempo_Parado,
    e.Tempo_Caminhando,
    e.Tempo_Correndo,
    e.Tempo_Sprintando,
    e.Coeficiente_Variacao_Velocidade,
    e.Quantidade_Alta_Aceleracao,
    e.Quantidade_Alta_Desaceleracao,
    e.Media_Curvatura
FROM Estatisticas_Segmento e
JOIN Segmentos s ON s.ID_Segmento = e.ID_Segmento
JOIN Jogadores j ON j.ID_Jogador = s.ID_Jogador
JOIN Jogos g ON g.ID_Jogo = s.ID_Jogo;

DROP VIEW IF EXISTS View_ResumoPorJogadorEPartida;
-- One row per (player, match). Aggregates the segment statistics directly, so
-- SUMs count each segment once. Total distance comes from a scalar subquery
-- over the trajectory points rather than a join, which is what inflated every
-- aggregate in the original definition.
CREATE VIEW View_ResumoPorJogadorEPartida AS
SELECT
    j.Nome_Jogador,
    j.Funcao_Jogador,
    g.Data_Jogo,
    g.Adversario,
    COUNT(*)                             AS Total_Segmentos,
    AVG(es.Velocidade_Media)             AS Velocidade_Media,
    AVG(es.Aceleracao_Media)             AS Aceleracao_Media,
    AVG(es.Coeficiente_Variacao_Velocidade) AS Coef_Variacao_Velocidade_Media,
    AVG(es.Media_Curvatura)              AS Curvatura_Media,
    SUM(es.Quantidade_Sprints)           AS Total_Sprints,
    SUM(es.Tempo_Parado)                 AS Tempo_Total_Parado,
    SUM(es.Tempo_Caminhando)             AS Tempo_Total_Caminhando,
    SUM(es.Tempo_Correndo)               AS Tempo_Total_Correndo,
    SUM(es.Tempo_Sprintando)             AS Tempo_Total_Sprintando,
    SUM(es.Quantidade_Alta_Aceleracao)   AS Total_Alta_Aceleracao,
    SUM(es.Quantidade_Alta_Desaceleracao) AS Total_Alta_Desaceleracao,
    (
        SELECT SUM(pt.Distancia)
        FROM Pontos_Trajetoria pt
        JOIN Segmentos s2 ON s2.ID_Segmento = pt.ID_Segmento
        WHERE s2.ID_Jogador = j.ID_Jogador AND s2.ID_Jogo = g.ID_Jogo
    )                                    AS Distancia_Total
FROM Estatisticas_Segmento es
JOIN Segmentos s  ON s.ID_Segmento = es.ID_Segmento
JOIN Jogadores j  ON j.ID_Jogador  = s.ID_Jogador
JOIN Jogos g      ON g.ID_Jogo     = s.ID_Jogo
GROUP BY j.ID_Jogador, g.ID_Jogo;
"""

TRAJECTORIES_CSV = "game_trajectories_processed.csv"
STATS_CSV = "stats_processed.csv"


def rebuild_views(database: Path, on_progress: Callable[[str], None] | None = None) -> None:
    """Replace the views on an existing database. Data is untouched."""
    conn = sqlite3.connect(database)
    try:
        conn.executescript(VIEWS)
        conn.commit()
    finally:
        conn.close()
    if on_progress:
        on_progress(f"rebuilt views on {database.name}")


def build(
    source: Path,
    target: Path,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Build ``soccer.sqlite`` from the processed trajectory and stats CSVs."""
    import pandas as pd

    def log(message: str) -> None:
        if on_progress:
            on_progress(message)

    trajectories_path = source / TRAJECTORIES_CSV
    stats_path = source / STATS_CSV
    for path in (trajectories_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path.name} in {source}. See docs/DATA.md -- the football "
                f"data is under a restricted-use agreement and is not distributed "
                f"with this repository."
            )

    log(f"reading {TRAJECTORIES_CSV}")
    trajectories = pd.read_csv(trajectories_path)
    log(f"reading {STATS_CSV}")
    stats = pd.read_csv(stats_path)

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    conn = sqlite3.connect(target)
    try:
        conn.executescript(SCHEMA)

        log("inserting matches and players")
        matches = trajectories[["Data_Jogo", "Adversário"]].drop_duplicates()
        conn.executemany(
            "INSERT OR IGNORE INTO Jogos (Data_Jogo, Adversario) VALUES (?, ?)",
            matches.itertuples(index=False, name=None),
        )
        players = trajectories[["Nome_Jogador", "Funcao_Jogador"]].drop_duplicates()
        conn.executemany(
            "INSERT OR IGNORE INTO Jogadores (Nome_Jogador, Funcao_Jogador) VALUES (?, ?)",
            players.itertuples(index=False, name=None),
        )
        conn.commit()

        match_ids = {
            (date, opponent): identifier
            for identifier, date, opponent in conn.execute(
                "SELECT ID_Jogo, Data_Jogo, Adversario FROM Jogos"
            )
        }
        player_ids = {
            name: identifier
            for identifier, name in conn.execute("SELECT ID_Jogador, Nome_Jogador FROM Jogadores")
        }

        log("inserting segments")
        # Renamed because `itertuples` mangles the accented column name into a
        # positional `_2`, which makes the loop below unreadable and fragile.
        segments = (
            trajectories[["Data_Jogo", "Adversário", "Nome_Jogador", "ID_Segmento", "Movimento"]]
            .drop_duplicates()
            .rename(columns={"Adversário": "Adversario"})
        )
        rows = [
            (
                match_ids[(row.Data_Jogo, row.Adversario)],
                player_ids[row.Nome_Jogador],
                row.ID_Segmento,
                row.Movimento,
            )
            for row in segments.itertuples(index=False)
        ]
        conn.executemany(
            "INSERT INTO Segmentos (ID_Jogo, ID_Jogador, ID_Segmento_Original, Movimento)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()

        segment_ids: dict[tuple, int] = {}
        for identifier, jogo, jogador, original in conn.execute(
            "SELECT ID_Segmento, ID_Jogo, ID_Jogador, ID_Segmento_Original FROM Segmentos"
        ):
            segment_ids[(jogo, jogador, original)] = identifier

        def resolve(frame: pd.DataFrame) -> list[int | None]:
            return [
                segment_ids.get(
                    (
                        match_ids.get((date, opponent)),
                        player_ids.get(player),
                        original,
                    )
                )
                for date, opponent, player, original in zip(
                    frame["Data_Jogo"],
                    frame["Adversário"],
                    frame["Nome_Jogador"],
                    frame["ID_Segmento"],
                    strict=True,
                )
            ]

        log(f"inserting {len(trajectories):,} trajectory points")
        trajectories = trajectories.assign(_sid=resolve(trajectories))
        point_columns = [
            "_sid",
            "Indice_Ponto",
            "Latitude_UTM",
            "Longitude_UTM",
            "Latitude_WGS84",
            "Longitude_WGS84",
            "Elevacao",
            "Timestamp",
            "Intervalo_Tempo",
            "Velocidade",
            "Direcao",
            "Distancia",
            "Zona_UTM",
        ]
        conn.executemany(
            "INSERT INTO Pontos_Trajetoria (ID_Segmento, Indice_Ponto, Latitude_UTM,"
            " Longitude_UTM, Latitude_WGS84, Longitude_WGS84, Elevacao, Timestamp,"
            " Intervalo_Tempo, Velocidade, Direcao, Distancia, Zona_UTM)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            trajectories[point_columns].itertuples(index=False, name=None),
        )
        conn.commit()

        log(f"inserting {len(stats):,} segment statistics")
        stats = stats.assign(_sid=resolve(stats))
        stats = stats[stats["_sid"].notna()]
        stat_columns = [
            "_sid",
            "Velocidade_Media",
            "Desvio_Padrao_Velocidade",
            "Aceleracao_Media",
            "Desvio_Aceleracao",
            "Quantidade_Sprints",
            "Media_Mudanca_Direcao",
            "Quantidade_Mudanca_Direcao",
            "Tempo_Parado",
            "Tempo_Caminhando",
            "Tempo_Correndo",
            "Tempo_Sprintando",
            "Coeficiente_Variacao_Velocidade",
            "Quantidade_Alta_Aceleracao",
            "Quantidade_Alta_Desaceleracao",
            "Media_Curvatura",
            "Desvio_Curvatura",
        ]
        conn.executemany(
            "INSERT INTO Estatisticas_Segmento (ID_Segmento, Velocidade_Media,"
            " Desvio_Velocidade, Aceleracao_Media, Desvio_Aceleracao, Quantidade_Sprints,"
            " Media_Mudanca_Direcao, Quantidade_Mudanca_Direcao, Tempo_Parado,"
            " Tempo_Caminhando, Tempo_Correndo, Tempo_Sprintando,"
            " Coeficiente_Variacao_Velocidade, Quantidade_Alta_Aceleracao,"
            " Quantidade_Alta_Desaceleracao, Media_Curvatura, Desvio_Curvatura)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            stats[stat_columns].itertuples(index=False, name=None),
        )
        conn.commit()

        log("creating views")
        conn.executescript(VIEWS)
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()

    log(f"done: {target}")
    return target
