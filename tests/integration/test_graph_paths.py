"""Runs the whole compiled graph with scripted models and fake Snowflake clients.

Every scenario is one path through the graph. `test_scenarios_cover_every_edge` checks that together they
walk every edge of the graph, so a new node or edge without a scenario fails the suite.
"""

from dataclasses import dataclass, field
from itertools import pairwise

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from us_population_agent.agent import graph as graph_module
from us_population_agent.agent.graph import build_graph
from us_population_agent.agent.nodes.answer import FALLBACK_NO_DATA
from us_population_agent.agent.nodes.no_data import FAILED_QUERY, UNAVAILABLE
from us_population_agent.agent.schemas import IntentDecision, QueryRewrite, SqlGeneration, TableSelection
from us_population_agent.config import Settings
from us_population_agent.snowflake_client.columns import Column, ColumnCatalogError, TableColumns
from us_population_agent.snowflake_client.query import QueryError, QueryResult, QueryUnavailableError
from us_population_agent.snowflake_client.search import TableHit

SETTINGS = Settings.model_construct(
    snowflake_census_database="CENSUS",
    snowflake_census_schema="PUBLIC",
    snowflake_database="APP",
    snowflake_schema="PUBLIC",
    snowflake_geography_table="V_GEOGRAPHY",
    max_repairs=2,
    max_search_queries=4,
    max_candidates=8,
)
GOOD_SQL = (
    'SELECT v.STATE_ABBR, SUM(d."B01001e1") AS TOTAL_POPULATION, COUNT(*) AS N_CBG '
    'FROM CENSUS.PUBLIC."2020_CBG_B01" d JOIN APP.PUBLIC.V_GEOGRAPHY v USING (CENSUS_BLOCK_GROUP) '
    "WHERE v.STATE_ABBR = 'TX' GROUP BY v.STATE_ABBR"
)
BAD_SQL = GOOD_SQL.replace("V_GEOGRAPHY", "OTHER")  # fails validation: table not allowed
RESULT = QueryResult(columns=["STATE_ABBR", "TOTAL_POPULATION", "N_CBG"], rows=[["TX", 29000000, 100]])
NO_ROWS = QueryResult(columns=["TOTAL_POPULATION"], rows=[])
HITS = [TableHit("B01001", "2020_CBG_B01", "Sex by age", "text", None)]
FOUND = TableColumns("2020_CBG_B01", "prefix", [Column("B01001e1", "Total")])
SELECT_B01001 = TableSelection(reason="fits", table_id="B01001")
NO_TABLE = TableSelection(reason="none fit", table_id=None)
INVENTED_TABLE = TableSelection(reason="r", table_id="INVENTED")
QUESTION = "How many people live in Texas?"


def generation(sql: str | None, notes: str = "") -> SqlGeneration:
    return SqlGeneration(selected_columns=[], sql=sql, unanswered=[], notes=notes)


GOOD = generation(GOOD_SQL)
BAD = generation(BAD_SQL)


class ScriptedLLM:
    """Chat model: plain `invoke` gives a text reply, structured output gives the next scripted value per schema.

    The last value of a script is repeated, so a single value means "always this".
    """

    def __init__(self, structured: dict[type, object], text: str | Exception = "model reply"):
        self.scripts = {schema: list(v) if isinstance(v, list) else [v] for schema, v in structured.items()}
        self.text = text

    def with_structured_output(self, schema):
        script = self.scripts[schema]

        class Structured:
            def invoke(self, messages):
                value = script.pop(0) if len(script) > 1 else script[0]
                if isinstance(value, Exception):
                    raise value
                return value

        return Structured()

    def invoke(self, messages):
        if isinstance(self.text, Exception):
            raise self.text
        return AIMessage(self.text)


class Scripted:
    """Fake client: every call takes the next outcome (the last one repeats) and returns or raises it."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list = []

    def next(self, call):
        self.calls.append(call)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Catalog(Scripted):
    def search(self, query):
        return self.next(query)


class Columns(Scripted):
    def columns_for(self, table_id):
        return self.next(table_id)


class Runner(Scripted):
    def run(self, sql):
        return self.next(sql)


@dataclass
class Scenario:
    """One question and the fakes it meets; the defaults describe the happy path."""

    intent: str = "answerable"
    catalog: tuple = (HITS,)
    columns: tuple = (FOUND,)
    selection: TableSelection = field(default_factory=lambda: SELECT_B01001)
    sql: tuple = (GOOD,)
    runner: tuple = (RESULT,)
    reply_text: str | Exception = "model reply"


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(graph_module, "get_settings", lambda: SETTINGS)


def make_graph(scenario: Scenario):
    llm = ScriptedLLM(
        {
            IntentDecision: IntentDecision(reason="r", intent=scenario.intent),
            QueryRewrite: QueryRewrite(notes="", queries=["population by age"]),
            TableSelection: scenario.selection,
        },
        text=scenario.reply_text,
    )
    sql_llm = ScriptedLLM({SqlGeneration: list(scenario.sql)})
    runner = Runner(*scenario.runner)
    compiled = build_graph(
        llm,
        Catalog(*scenario.catalog),
        Columns(*scenario.columns),
        runner,
        sql_llm,
        SETTINGS,
        InMemorySaver(),
    )
    return compiled, runner


def ask(compiled, text: str = QUESTION, thread: str = "t") -> tuple[list[str], str]:
    """Send one message; return the nodes visited in order and the final reply."""
    config = {"configurable": {"thread_id": thread}}
    visited: list[str] = []
    for update in compiled.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
        visited.extend(update)
    return visited, compiled.get_state(config).values["messages"][-1].text


ANSWER = ["intent", "rewrite", "search", "select_tables", "load_columns", "generate_sql", "validate_sql"]
GENERATED = ["generate_sql", "validate_sql"]
REPAIRED = ["repair_sql", "validate_sql"]
UNTIL_TABLE = ["intent", "rewrite", "search", "select_tables"]
UNTIL_SQL = [*UNTIL_TABLE, "load_columns"]

# id -> (scenario, nodes visited, final reply; None when the reply is the model's)
SCENARIOS: dict[str, tuple[Scenario, list[str], str | None]] = {
    "answer": (Scenario(), [*ANSWER, "execute_sql", "answer"], "model reply"),
    "answer_with_no_rows": (Scenario(runner=(NO_ROWS,)), [*ANSWER, "execute_sql", "answer"], "model reply"),
    "clarify": (Scenario(intent="needs_clarification"), ["intent", "clarify"], "model reply"),
    "meta": (Scenario(intent="meta"), ["intent", "meta"], "model reply"),
    "off_topic": (Scenario(intent="off_topic"), ["intent", "off_topic"], "model reply"),
    "search_outage": (
        Scenario(catalog=(RuntimeError("down"),)),
        ["intent", "rewrite", "search", "unavailable"],
        UNAVAILABLE,
    ),
    "no_candidates": (Scenario(catalog=([],), selection=NO_TABLE), [*UNTIL_TABLE, "no_table"], "model reply"),
    "no_fitting_table": (Scenario(selection=NO_TABLE), [*UNTIL_TABLE, "no_table"], "model reply"),
    "invented_table": (Scenario(selection=INVENTED_TABLE), [*UNTIL_TABLE, "no_table"], "model reply"),
    "unknown_table_columns": (Scenario(columns=(None,)), [*UNTIL_SQL, "unavailable"], UNAVAILABLE),
    "columns_lookup_fails": (Scenario(columns=(ColumnCatalogError("x"),)), [*UNTIL_SQL, "unavailable"], UNAVAILABLE),
    "columns_unexpected_error": (Scenario(columns=(RuntimeError("x"),)), [*UNTIL_SQL, "unavailable"], UNAVAILABLE),
    "model_writes_no_sql": (
        Scenario(sql=(generation(None, "no column"),)),
        [*UNTIL_SQL, "generate_sql", "no_sql"],
        "model reply",
    ),
    "model_fails_writing_sql": (
        Scenario(sql=(RuntimeError("boom"),)),
        [*UNTIL_SQL, "generate_sql", "unavailable"],
        UNAVAILABLE,
    ),
    "invalid_sql_repaired": (
        Scenario(sql=(BAD, GOOD)),
        [*ANSWER, *REPAIRED, "execute_sql", "answer"],
        "model reply",
    ),
    "invalid_sql_repaired_twice": (
        Scenario(sql=(BAD, BAD, GOOD)),
        [*ANSWER, *REPAIRED, *REPAIRED, "execute_sql", "answer"],
        "model reply",
    ),
    "invalid_sql_never_fixed": (
        Scenario(sql=(BAD,)),
        [*ANSWER, *REPAIRED, *REPAIRED, "failed_query"],
        FAILED_QUERY,
    ),
    "repair_gives_up": (
        Scenario(sql=(BAD, generation(None, "cannot fix"))),
        [*ANSWER, "repair_sql", "no_sql"],
        "model reply",
    ),
    "repair_model_fails": (
        Scenario(sql=(BAD, RuntimeError("boom"))),
        [*ANSWER, "repair_sql", "unavailable"],
        UNAVAILABLE,
    ),
    "snowflake_rejects_then_repair_works": (
        Scenario(sql=(GOOD, GOOD), runner=(QueryError("syntax"), RESULT)),
        [*ANSWER, "execute_sql", *REPAIRED, "execute_sql", "answer"],
        "model reply",
    ),
    "snowflake_always_rejects": (
        Scenario(runner=(QueryError("syntax"),)),
        [*ANSWER, "execute_sql", *REPAIRED, "execute_sql", *REPAIRED, "execute_sql", "failed_query"],
        FAILED_QUERY,
    ),
    "validation_then_snowflake_error": (
        Scenario(sql=(BAD, GOOD, GOOD), runner=(QueryError("syntax"), RESULT)),
        [*ANSWER, *REPAIRED, "execute_sql", *REPAIRED, "execute_sql", "answer"],
        "model reply",
    ),
    "snowflake_outage": (
        Scenario(runner=(QueryUnavailableError("down"),)),
        [*ANSWER, "execute_sql", "unavailable"],
        UNAVAILABLE,
    ),
    "snowflake_unexpected_error": (
        Scenario(runner=(OSError("connection reset"),)),
        [*ANSWER, "execute_sql", "unavailable"],
        UNAVAILABLE,
    ),
    "snowflake_outage_after_repair": (
        Scenario(sql=(BAD, GOOD), runner=(QueryUnavailableError("down"),)),
        [*ANSWER, *REPAIRED, "execute_sql", "unavailable"],
        UNAVAILABLE,
    ),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_path(name):
    scenario, expected_path, expected_reply = SCENARIOS[name]
    compiled, _ = make_graph(scenario)
    visited, reply = ask(compiled)
    assert visited == expected_path
    assert reply == (expected_reply or "model reply")


def test_query_reaches_snowflake_only_after_validation():
    compiled, runner = make_graph(SCENARIOS["invalid_sql_repaired"][0])
    ask(compiled)
    assert runner.calls == [GOOD_SQL]


def test_repairs_are_limited_to_the_configured_number():
    compiled, runner = make_graph(SCENARIOS["snowflake_always_rejects"][0])
    ask(compiled)
    assert len(runner.calls) == 1 + SETTINGS.max_repairs


def test_outage_is_not_retried():
    compiled, runner = make_graph(SCENARIOS["snowflake_outage"][0])
    ask(compiled)
    assert len(runner.calls) == 1


def test_non_data_intents_never_touch_snowflake():
    for name in ("clarify", "meta", "off_topic"):
        compiled, runner = make_graph(SCENARIOS[name][0])
        ask(compiled)
        assert runner.calls == []


def test_failing_reply_model_still_gives_a_message():
    compiled, _ = make_graph(Scenario(reply_text=RuntimeError("boom")))
    _, reply = ask(compiled)
    assert reply.startswith("Here is the result:")


def test_failing_reply_model_without_rows_gives_the_no_data_message():
    compiled, _ = make_graph(Scenario(runner=(NO_ROWS,), reply_text=RuntimeError("boom")))
    assert ask(compiled)[1] == FALLBACK_NO_DATA


def test_failure_in_one_turn_does_not_leak_into_the_next():
    # turn 1: Snowflake is down; turn 2 on the same thread: it works again
    compiled, _ = make_graph(Scenario(runner=(QueryUnavailableError("down"), RESULT)))
    assert ask(compiled)[1] == UNAVAILABLE
    visited, reply = ask(compiled, "And in Ohio?")
    assert visited == [*ANSWER, "execute_sql", "answer"]
    assert reply == "model reply"


def test_search_outage_does_not_leak_into_the_next_turn():
    compiled, _ = make_graph(Scenario(catalog=(RuntimeError("down"), HITS)))
    assert ask(compiled)[0][-1] == "unavailable"
    assert ask(compiled)[0][-1] == "answer"


def test_repair_counter_starts_over_every_turn():
    # each turn needs one repair; with a counter that leaked, the second turn would run out of attempts
    compiled, _ = make_graph(Scenario(sql=(BAD, GOOD, BAD, GOOD)))
    for _ in range(2):
        visited, _reply = ask(compiled)
        assert visited[-1] == "answer"


def test_history_is_kept_per_thread():
    compiled, _ = make_graph(Scenario())
    ask(compiled, "first", thread="a")
    ask(compiled, "second", thread="a")
    ask(compiled, "other", thread="b")
    texts = {
        thread: [m.text for m in compiled.get_state({"configurable": {"thread_id": thread}}).values["messages"]]
        for thread in "ab"
    }
    assert texts["a"] == ["first", "model reply", "second", "model reply"]
    assert texts["b"] == ["other", "model reply"]


def test_scenarios_cover_every_edge():
    compiled, _ = make_graph(Scenario())
    edges = {(edge.source, edge.target) for edge in compiled.get_graph().edges}
    walked = set()
    for _, path, _ in SCENARIOS.values():
        walked |= {("__start__", path[0]), (path[-1], "__end__")}
        walked |= set(pairwise(path))
    assert edges - walked == set(), "edges no scenario walks"
    assert walked - edges == set(), "scenario path that is not an edge of the graph"
    nodes = {node for node in compiled.get_graph().nodes} - {"__start__", "__end__"}
    assert nodes == {node for _, path, _ in SCENARIOS.values() for node in path}
