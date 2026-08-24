"""Primitive SQL tools, exposed directly to the outer Reason-Act loop.

This is the fix for the structural finding in ``docs/REVIEW.md``: in the thesis
implementation the single available tool was itself a complete ReAct agent
(``create_react_agent`` over ``SQLDatabaseToolkit``), so it listed tables, read
schemas, wrote SQL, read results and retried *internally*. By the time it
returned there was nothing left for the outer loop to do, and 42 of the 47
surviving traces show exactly two reasoning iterations: reason, one tool call,
reason, answer.

With primitives, the interleaving the ReAct paradigm is actually about happens
in the graph the thesis draws in Figure 5:

    Thought: preciso saber quais tabelas existem
    Action: listar_tabelas                  -> Observation: Jogos, Jogadores, ...
    Thought: preciso do schema de Estatisticas_Segmento
    Action: descrever_tabelas               -> Observation: CREATE TABLE ...
    Thought: agora posso montar a consulta
    Action: executar_sql                    -> Observation: {...}
    Thought: tenho a resposta

Two further consequences, both good:

* **no second language model.** These tools are plain SQLite calls, so sweeping
  the agent model no longer sweeps a hidden SQL-generating model with it
  (``ERRATA.md``, SV-08 -- solved rather than merely made configurable);
* **units survive.** ``executar_sql`` annotates result columns with the unit
  declared in the domain config, so an observation reads
  ``{"Velocidade_Media (m/s)": 1.36}`` instead of a bare number. The Tailor step
  can then be given observations alone without losing the unit -- which is what
  made ``tailor_sees: observations`` produce "1,36 (naquela unidade de medida
  do time)" while the nested agent was in use.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import DomainConfig

MAX_ROWS = 50
MAX_CHARS = 4000

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


class SQLToolError(RuntimeError):
    pass


def annotate_units(row: dict[str, Any], units: dict[str, str]) -> dict[str, Any]:
    """Append the declared unit to each column name.

    ``{"Velocidade_Media": 1.36}`` becomes ``{"Velocidade_Media (m/s)": 1.36}``.

    The unit is the single most fragile piece of information in this pipeline:
    a legacy answer reports "1.362 km/h" for a column stored in metres per
    second. Carrying the unit *in the observation* means every downstream step
    -- the Reason node, the Tailor, and the factuality check -- sees it.
    """
    if not units:
        return row
    out: dict[str, Any] = {}
    for column, value in row.items():
        unit = units.get(column)
        out[f"{column} ({unit})" if unit else column] = value
    return out


@dataclass
class SQLPrimitives:
    """The three tools, sharing one read-only connection."""

    domain: DomainConfig
    calls: list[dict[str, Any]] = field(default_factory=list)
    _conn: sqlite3.Connection | None = field(default=None, repr=False)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            path = self.domain.database_path
            if not path.is_file():
                raise SQLToolError(f"database missing: {path}")
            # Read-only: the agent cannot damage the data even if the model
            # emits DML that slips past the guard below.
            self._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ----------------------------------------------------------------- tools

    def list_tables(self, _: str = "") -> str:
        """Names of every table and view in the database."""
        conn = self._connection()
        rows = conn.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type, name"
        ).fetchall()
        tables = [r["name"] for r in rows if r["type"] == "table"]
        views = [r["name"] for r in rows if r["type"] == "view"]
        parts = [f"Tabelas: {', '.join(tables)}"]
        if views:
            parts.append(f"Views: {', '.join(views)}")
        return "\n".join(parts)

    def describe_tables(self, names: str) -> str:
        """The CREATE statement for one or more comma-separated tables or views."""
        conn = self._connection()
        wanted = [n.strip() for n in re.split(r"[,\s]+", names or "") if n.strip()]
        if not wanted:
            return "[Informe ao menos um nome de tabela ou view.]"

        known = {
            r["name"]: r["sql"]
            for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL")
        }
        out: list[str] = []
        for name in wanted:
            if name not in known:
                close = [k for k in known if k.lower() == name.lower()]
                if close:
                    name = close[0]
                else:
                    out.append(f"-- {name}: não existe. Use listar_tabelas.")
                    continue
            out.append(known[name])
        return _truncate("\n\n".join(out))

    def run_sql(self, query: str) -> str:
        """Execute a read-only SELECT and return the rows as JSON."""
        query = (query or "").strip().rstrip(";")
        if not query:
            return "[Consulta vazia.]"
        if _FORBIDDEN.search(query):
            return "[Apenas consultas de leitura são permitidas. Use SELECT ou WITH ... SELECT.]"
        if not re.match(r"^\s*(SELECT|WITH)\b", query, re.IGNORECASE):
            return "[A consulta deve começar com SELECT ou WITH.]"

        conn = self._connection()
        try:
            cursor = conn.execute(query)
            rows = cursor.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as exc:
            # Returned, not raised: a SQL error is an observation the agent can
            # reason about and correct on the next iteration. That correction
            # loop is the whole point of exposing primitives.
            return f"[Erro de SQL: {exc}]"

        truncated = len(rows) > MAX_ROWS
        payload = [annotate_units(dict(r), self.domain.units) for r in rows[:MAX_ROWS]]
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if truncated:
            text += f"\n[Resultado truncado em {MAX_ROWS} linhas.]"
        return _truncate(text)

    # ------------------------------------------------------------- registry

    def as_registry(self) -> dict[str, Any]:
        return {
            "listar_tabelas": self.list_tables,
            "descrever_tabelas": self.describe_tables,
            "executar_sql": self.run_sql,
        }

    def __call__(self, name: str, argument: str) -> str:
        """Dispatch by tool name, recording the call for the audit trail."""
        started = time.perf_counter()
        record: dict[str, Any] = {"tool": name, "input": argument, "queries": [], "ok": True}
        try:
            fn = self.as_registry().get(name)
            if fn is None:
                record["ok"] = False
                return f"[Ferramenta desconhecida: {name!r}]"
            if name == "executar_sql":
                record["queries"].append(argument)
            output = fn(argument)
            record["output"] = output
            record["ok"] = not output.startswith("[Erro")
            return output
        finally:
            record["elapsed_s"] = round(time.perf_counter() - started, 3)
            self.calls.append(record)


def _truncate(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    return text[:MAX_CHARS] + f"\n[Saída truncada em {MAX_CHARS} caracteres.]"


TOOL_DESCRIPTIONS = {
    "listar_tabelas": "Lista todas as tabelas e views do banco. Não recebe argumento.",
    "descrever_tabelas": (
        "Mostra o comando CREATE de uma ou mais tabelas/views. "
        "Argumento: nomes separados por vírgula."
    ),
    "executar_sql": (
        "Executa uma consulta SELECT no banco e devolve as linhas em JSON. "
        "Argumento: a consulta SQL."
    ),
}
