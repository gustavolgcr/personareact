"""The Text-to-SQL tool: a natural-language question against a SQLite database.

Three fixes relative to the original ``tools.py``:

* **the tool receives the agent's ``action_input``.** The original had
  ``inp = state.question if state.action == "consulta_banco_sql" else state.action_input``
  and ``consulta_banco_sql`` was the only tool ever registered, so the branch
  that always fired discarded whatever the Reason step had decided to look up
  and re-sent the raw original question. Every iteration of the loop therefore
  issued the same query, which made multi-step retrieval impossible.

* **the SQL model is separable from the agent model.** The original created a
  module-level ``ChatOpenAI`` with the same global ``OPENAI_MODEL``, so
  sweeping the model swept the SQL generator too and the cross-model comparison
  conflated personalization ability with text-to-SQL ability.

* **the executed SQL is captured.** Every query and its result is recorded on
  the state, so an answer can be audited against the database afterwards.

The tool is still internally an agent (``create_react_agent`` over
``SQLDatabaseToolkit``), which is a real confound worth naming: a share of the
reasoning happens inside the tool rather than in the measured outer loop. That
is left as-is deliberately so the comparison with the thesis holds, and is
documented in ``docs/ERRATA.md`` (SV-08).
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..config import DomainConfig, ModelConfig
from ..llm import Usage
from ..manifest import sha256_file
from ..prompts import render

_SQL_IN_TEXT = re.compile(r"\b(SELECT|WITH)\b.+?(?:;|$)", re.IGNORECASE | re.DOTALL)

#: Prefix of the observation returned when the tool fails. A failed lookup is
#: still an observation the agent can react to, so it is returned rather than
#: raised -- but it must never be mistaken for an answer downstream, which is
#: what :func:`is_tool_error` is for.
TOOL_ERROR_PREFIX = "[Erro ao executar consulta]"


def is_tool_error(text: str) -> bool:
    return bool(text) and text.lstrip().startswith(TOOL_ERROR_PREFIX)


class DatabaseError(RuntimeError):
    pass


def verify_database(domain: DomainConfig, *, strict: bool = True) -> dict[str, Any]:
    """Check that the database is present, complete and the expected build.

    This exists because of a defect the audit found in the original tree: the
    soccer database had been copied into nine project folders and had drifted
    into three distinct builds. PersonaReAct queried a build from 24 Apr 2025
    with no SQL views; the ReAct baselines queried a build from 09 May 2025
    carrying ``View_EstatisticasComJogadorJogo`` and
    ``View_ResumoPorJogadorEPartida``. Both sides passed ``view_support=True``,
    so the two conditions were handed structurally different schemas -- the
    baselines got a ready-made per-player-per-match aggregate, PersonaReAct had
    to write the JOIN and GROUP BY itself.

    One database, checksummed, verified before every run.
    """
    path = domain.database_path
    if not path.is_file():
        raise DatabaseError(
            f"database missing: {path}\n"
            f"Build it with:  personareact data build --domain {domain.name}"
        )

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        views = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        conn.close()

    missing_views = [v for v in domain.expects_views if v not in views]
    # Always computed, so `data verify --update` can pin a database that has
    # none declared yet. Only *compared* when the config declares one.
    digest = sha256_file(path)
    report: dict[str, Any] = {
        "path": str(path),
        "tables": sorted(tables),
        "views": sorted(views),
        "missing_views": missing_views,
        "sha256": digest,
        "sha256_ok": None,
    }

    if domain.database_sha256:
        report["sha256_ok"] = digest == domain.database_sha256
        if strict and not report["sha256_ok"]:
            raise DatabaseError(
                f"database checksum mismatch for {domain.name}\n"
                f"  expected {domain.database_sha256}\n"
                f"  found    {digest}\n"
                f"This is exactly the failure mode that made the thesis runs "
                f"non-comparable. Rebuild with `personareact data build`, or "
                f"update database_sha256 in configs/domain/{domain.name}.yaml "
                f"if the new build is intentional."
            )

    if strict and missing_views:
        raise DatabaseError(
            f"database for {domain.name} is missing expected view(s): "
            f"{', '.join(missing_views)}. Rebuild with `personareact data build`."
        )

    return report


@dataclass
class TextToSQLTool:
    """Natural-language question in, textual answer out."""

    domain: DomainConfig
    model: ModelConfig
    api_key: str | None = None
    verbose: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    #: Tokens and cost spent inside the nested Text-to-SQL agent. Counted
    #: separately from the outer agent because it is a *different* model
    #: (``sql_model``) and because it is by far the larger consumer -- the
    #: nested agent lists tables, reads schemas, then queries. Leaving it
    #: uncounted made a run look about five times cheaper than it was.
    usage: Usage = field(default_factory=Usage)

    name: str = "consulta_banco_sql"
    _agent: Any = field(default=None, repr=False)
    _db: SQLDatabase | None = field(default=None, repr=False)

    @property
    def description(self) -> str:
        return (
            "Recebe uma pergunta em português (linguagem natural), converte "
            "internamente em SQL, executa no banco de dados e retorna a resposta "
            "em texto."
        )

    def _build(self) -> Any:
        if self._agent is None:
            path: Path = self.domain.database_path
            self._db = SQLDatabase.from_uri(f"sqlite:///{path}", view_support=True)
            kwargs: dict[str, Any] = {"model": self.model.effective_sql_model}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            kwargs.update(self.model.sampling_kwargs())
            llm = ChatOpenAI(**kwargs)
            toolkit = SQLDatabaseToolkit(db=self._db, llm=llm)
            prompt = render(
                "text_to_sql",
                scope=self.domain.prompt_dir,
                dialect="SQLite",
            )
            self._agent = create_react_agent(llm, toolkit.get_tools(), prompt=prompt)
        return self._agent

    def _record_usage(self, message: Any) -> None:
        meta = getattr(message, "usage_metadata", None) or {}
        inp = int(meta.get("input_tokens", 0) or 0)
        out = int(meta.get("output_tokens", 0) or 0)
        if not (inp or out):
            return
        self.usage.input_tokens += inp
        self.usage.output_tokens += out
        self.usage.usd += (
            inp * self.model.usd_per_1m_input + out * self.model.usd_per_1m_output
        ) / 1_000_000
        self.usage.calls += 1

    def __call__(self, question: str) -> str:
        started = time.perf_counter()
        record: dict[str, Any] = {"input": question, "queries": [], "ok": True}
        try:
            agent = self._build()
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            messages = result.get("messages", [])
            if not messages:
                raise DatabaseError("Text-to-SQL agent returned no messages")

            before = self.usage.usd
            for message in messages:
                for query in _extract_queries(message):
                    record["queries"].append(query)
                self._record_usage(message)
                if self.verbose:
                    printer = getattr(message, "pretty_print", None)
                    if printer:
                        printer()

            answer = getattr(messages[-1], "content", "") or ""
            record["output"] = answer
            record["usd"] = round(self.usage.usd - before, 6)
            return answer
        except Exception as exc:
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["output"] = ""
            return f"{TOOL_ERROR_PREFIX}: {exc}"
        finally:
            record["elapsed_s"] = round(time.perf_counter() - started, 3)
            self.calls.append(record)


def _extract_queries(message: Any) -> list[str]:
    """Pull the SQL out of a message's tool calls, for the audit trail.

    Only tool *calls* are inspected. Tool *results* are data, and a result row
    can easily contain the word SELECT without being a query.
    """
    found: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        args = call.get("args") or {}
        query = args.get("query") or args.get("__arg1")
        if isinstance(query, str) and _SQL_IN_TEXT.search(query):
            found.append(query.strip())
    return found
