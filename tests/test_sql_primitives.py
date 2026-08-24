"""Tests for the primitive SQL tools.

These exist to fix the structural finding in ``docs/REVIEW.md``: with a nested
ReAct agent as the only tool, the outer Reason-Act loop ran exactly twice in 42
of 47 surviving traces. Primitives put the interleaving back in the graph the
thesis draws.
"""

from __future__ import annotations

import sqlite3

import pytest

from personareact.config import DomainConfig
from personareact.tools.sql_primitives import SQLPrimitives, annotate_units


@pytest.fixture
def primitives(tmp_path, monkeypatch):
    database = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(database)
    conn.executescript(
        """
        CREATE TABLE Jogadores (ID_Jogador INTEGER PRIMARY KEY, Nome_Jogador TEXT);
        CREATE TABLE Estatisticas_Segmento (
            ID_Jogador INTEGER, Velocidade_Media REAL, Tempo_Parado INTEGER
        );
        CREATE VIEW Resumo AS SELECT Nome_Jogador FROM Jogadores;
        INSERT INTO Jogadores VALUES (1, 'Bruno Pacheco');
        INSERT INTO Estatisticas_Segmento VALUES (1, 1.3622634444889454, 463);
        """
    )
    conn.commit()
    conn.close()

    domain = DomainConfig.load("soccer")
    object.__setattr__(domain, "database", database.name)
    monkeypatch.setattr("personareact.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(type(domain), "database_path", property(lambda self: database))
    return SQLPrimitives(domain=domain)


class TestUnits:
    def test_columns_are_annotated(self):
        units = {"Velocidade_Media": "m/s"}
        assert annotate_units({"Velocidade_Media": 1.36}, units) == {"Velocidade_Media (m/s)": 1.36}

    def test_unknown_columns_pass_through(self):
        assert annotate_units({"Nome": "x"}, {"Velocidade_Media": "m/s"}) == {"Nome": "x"}

    def test_no_units_is_a_no_op(self):
        assert annotate_units({"a": 1}, {}) == {"a": 1}

    def test_the_unit_reaches_the_observation(self, primitives):
        """The fix for the m/s versus km/h defect.

        A legacy answer reports "1.362 km/h" for a column stored in metres per
        second. If the unit is in the observation, no downstream step has to
        guess it -- which is what let `tailor_sees: observations` lose it.
        """
        out = primitives.run_sql("SELECT Velocidade_Media FROM Estatisticas_Segmento")
        assert "Velocidade_Media (m/s)" in out
        assert "1.3622634444889454" in out


class TestListAndDescribe:
    def test_tables_and_views_are_listed(self, primitives):
        out = primitives.list_tables()
        assert "Jogadores" in out and "Estatisticas_Segmento" in out
        assert "Resumo" in out

    def test_internal_tables_are_hidden(self, primitives):
        assert "sqlite_" not in primitives.list_tables()

    def test_describe_returns_the_create_statement(self, primitives):
        out = primitives.describe_tables("Jogadores")
        assert "CREATE TABLE Jogadores" in out

    def test_describe_accepts_several_names(self, primitives):
        out = primitives.describe_tables("Jogadores, Estatisticas_Segmento")
        assert "CREATE TABLE Jogadores" in out
        assert "CREATE TABLE Estatisticas_Segmento" in out

    def test_an_unknown_table_is_an_observation_not_a_crash(self, primitives):
        assert "não existe" in primitives.describe_tables("Inexistente")

    def test_describe_is_case_insensitive(self, primitives):
        assert "CREATE TABLE Jogadores" in primitives.describe_tables("jogadores")


class TestRunSQL:
    def test_a_select_returns_json(self, primitives):
        assert "Bruno Pacheco" in primitives.run_sql("SELECT Nome_Jogador FROM Jogadores")

    @pytest.mark.parametrize(
        "query",
        [
            "DROP TABLE Jogadores",
            "DELETE FROM Jogadores",
            "UPDATE Jogadores SET Nome_Jogador='x'",
            "INSERT INTO Jogadores VALUES (2,'x')",
            "PRAGMA table_info(Jogadores)",
        ],
    )
    def test_writes_are_refused(self, primitives, query):
        assert "leitura" in primitives.run_sql(query)

    def test_a_non_select_is_refused(self, primitives):
        assert "SELECT ou WITH" in primitives.run_sql("EXPLAIN SELECT 1")

    def test_a_cte_is_allowed(self, primitives):
        assert "total" in primitives.run_sql("WITH x AS (SELECT 1 AS total) SELECT * FROM x")

    def test_a_sql_error_comes_back_as_an_observation(self, primitives):
        """The agent must be able to read the error and correct itself.

        That correction loop is the entire point of exposing primitives, so a
        bad query has to be an observation rather than an exception.
        """
        out = primitives.run_sql("SELECT * FROM Inexistente")
        assert out.startswith("[Erro de SQL:")

    def test_the_database_is_opened_read_only(self, primitives):
        primitives.list_tables()
        with pytest.raises(sqlite3.OperationalError):
            primitives._connection().execute("CREATE TABLE Nova (x INTEGER)")

    def test_an_empty_query_is_handled(self, primitives):
        assert "vazia" in primitives.run_sql("   ")


class TestDispatch:
    def test_every_tool_is_reachable_by_name(self, primitives):
        assert set(primitives.as_registry()) == {
            "listar_tabelas",
            "descrever_tabelas",
            "executar_sql",
        }

    def test_calls_are_recorded_for_the_audit_trail(self, primitives):
        primitives("listar_tabelas", "")
        primitives("executar_sql", "SELECT 1 AS n")
        assert [c["tool"] for c in primitives.calls] == ["listar_tabelas", "executar_sql"]
        assert primitives.calls[1]["queries"] == ["SELECT 1 AS n"]

    def test_a_failed_query_is_flagged_in_the_record(self, primitives):
        primitives("executar_sql", "SELECT * FROM Inexistente")
        assert primitives.calls[-1]["ok"] is False

    def test_an_unknown_tool_name_is_an_observation(self, primitives):
        assert "desconhecida" in primitives("inventada", "x")


class TestActNodeArguments:
    """The Reason step's chosen argument reaches the tool unaltered (SV-07)."""

    @staticmethod
    def runtime(primitives):
        from personareact.agent.nodes import AgentRuntime
        from personareact.config import ApproachConfig, DomainConfig, ModelConfig
        from personareact.llm import LLMClient

        return AgentRuntime(
            approach=ApproachConfig.load("personareact"),
            domain=DomainConfig.load("soccer"),
            llm=LLMClient(config=ModelConfig.load("gpt-4.1-mini"), api_key="test"),
            primitives=primitives,
        )

    def test_an_empty_argument_stays_empty_for_a_primitive(self, primitives):
        """`listar_tabelas` takes no argument.

        Substituting the question would put the whole question in the audit
        trail as if it had been the chosen input -- the same class of defect as
        the original `inp = state.question if ... else state.action_input`.
        """
        from personareact.agent.nodes import make_act_node
        from personareact.agent.state import AgentState

        node = make_act_node(self.runtime(primitives))
        state = AgentState(question="pergunta original", action="listar_tabelas", action_input="")
        node(state)
        assert primitives.calls[-1]["input"] == ""

    def test_a_chosen_argument_is_passed_through(self, primitives):
        from personareact.agent.nodes import make_act_node
        from personareact.agent.state import AgentState

        node = make_act_node(self.runtime(primitives))
        state = AgentState(
            question="pergunta original",
            action="executar_sql",
            action_input="SELECT 1 AS n",
        )
        result = node(state)
        assert primitives.calls[-1]["input"] == "SELECT 1 AS n"
        assert "Observation:" in result["scratchpad"][-1]

    def test_an_unknown_action_becomes_an_observation(self, primitives):
        from personareact.agent.nodes import make_act_node
        from personareact.agent.state import AgentState

        node = make_act_node(self.runtime(primitives))
        result = node(AgentState(question="q", action="inventada", action_input="x"))
        assert "desconhecida" in result["observation"]
