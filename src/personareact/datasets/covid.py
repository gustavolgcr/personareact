"""Build the COVID-19 hospital-records database.

Source: the FAPESP COVID-19 Data Sharing/BR repository, Hospital Sírio-Libanês
subset. Public; ``docs/DATA.md`` has the download procedure.

Ported from the original ``create_database_covid.py`` with the same three
tables and the same pipe-separated parsing. Changes: bulk inserts instead of
row-at-a-time, indexes on the join keys, and the file is written straight to
``data/covid.sqlite`` rather than to ``hsl_dados.db`` in the working directory
followed by an undocumented manual rename (the original left both copies on
disk, 484 MB each, byte-identical).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS Pacientes (
    ID_PACIENTE TEXT PRIMARY KEY,
    IC_SEXO TEXT,
    AA_NASCIMENTO TEXT,
    CD_PAIS TEXT,
    CD_UF TEXT,
    CD_MUNICIPIO TEXT,
    CD_CEPREDUZIDO TEXT
);

CREATE TABLE IF NOT EXISTS Exames (
    ID_EXAME INTEGER PRIMARY KEY AUTOINCREMENT,
    ID_PACIENTE TEXT,
    ID_ATENDIMENTO TEXT,
    DT_COLETA TEXT,
    DE_ORIGEM TEXT,
    DE_EXAME TEXT,
    DE_ANALITO TEXT,
    DE_RESULTADO TEXT,
    CD_UNIDADE TEXT,
    DE_VALOR_REFERENCIA TEXT,
    FOREIGN KEY(ID_PACIENTE) REFERENCES Pacientes(ID_PACIENTE)
);

CREATE TABLE IF NOT EXISTS Desfechos (
    ID_DESFECHO INTEGER PRIMARY KEY AUTOINCREMENT,
    ID_PACIENTE TEXT,
    ID_ATENDIMENTO TEXT,
    DT_ATENDIMENTO TEXT,
    DE_TIPO_ATENDIMENTO TEXT,
    ID_CLINICA TEXT,
    DE_CLINICA TEXT,
    DT_DESFECHO TEXT,
    DE_DESFECHO TEXT,
    FOREIGN KEY(ID_PACIENTE) REFERENCES Pacientes(ID_PACIENTE)
);

-- Exames and Desfechos join on the PAIR (ID_PACIENTE, ID_ATENDIMENTO);
-- ID_ATENDIMENTO alone is not unique across patients.
CREATE INDEX IF NOT EXISTS idx_exames_paciente ON Exames(ID_PACIENTE);
CREATE INDEX IF NOT EXISTS idx_exames_atendimento ON Exames(ID_PACIENTE, ID_ATENDIMENTO);
CREATE INDEX IF NOT EXISTS idx_exames_analito ON Exames(DE_ANALITO);
CREATE INDEX IF NOT EXISTS idx_desfechos_paciente ON Desfechos(ID_PACIENTE);
CREATE INDEX IF NOT EXISTS idx_desfechos_atendimento ON Desfechos(ID_PACIENTE, ID_ATENDIMENTO);
"""

FILES = {
    "Pacientes": "HSL_Pacientes_4.csv",
    "Desfechos": "HSL_Desfechos_4.csv",
    "Exames": "HSL_Exames_4.csv",
}

COLUMNS = {
    "Pacientes": [
        "ID_PACIENTE",
        "IC_SEXO",
        "AA_NASCIMENTO",
        "CD_PAIS",
        "CD_UF",
        "CD_MUNICIPIO",
        "CD_CEPREDUZIDO",
    ],
    "Exames": [
        "ID_PACIENTE",
        "ID_ATENDIMENTO",
        "DT_COLETA",
        "DE_ORIGEM",
        "DE_EXAME",
        "DE_ANALITO",
        "DE_RESULTADO",
        "CD_UNIDADE",
        "DE_VALOR_REFERENCIA",
    ],
    "Desfechos": [
        "ID_PACIENTE",
        "ID_ATENDIMENTO",
        "DT_ATENDIMENTO",
        "DE_TIPO_ATENDIMENTO",
        "ID_CLINICA",
        "DE_CLINICA",
        "DT_DESFECHO",
        "DE_DESFECHO",
    ],
}


def build(
    source: Path,
    target: Path,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    import pandas as pd

    def log(message: str) -> None:
        if on_progress:
            on_progress(message)

    missing = [name for name in FILES.values() if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {', '.join(missing)} in {source}. See docs/DATA.md for the "
            f"download procedure (the dataset is public but requires registration)."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    conn = sqlite3.connect(target)
    try:
        conn.executescript(SCHEMA)

        for table, filename in FILES.items():
            log(f"reading {filename}")
            frame = pd.read_csv(source / filename, dtype=str, sep="|")
            frame = frame.reindex(columns=COLUMNS[table])
            placeholders = ",".join("?" * len(COLUMNS[table]))
            verb = "INSERT OR IGNORE" if table == "Pacientes" else "INSERT"
            log(f"inserting {len(frame):,} rows into {table}")
            conn.executemany(
                f"{verb} INTO {table} ({','.join(COLUMNS[table])}) VALUES ({placeholders})",
                frame.itertuples(index=False, name=None),
            )
            conn.commit()

        conn.execute("VACUUM")
    finally:
        conn.close()

    log(f"done: {target}")
    return target
