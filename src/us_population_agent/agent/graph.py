from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from us_population_agent.agent.nodes.answer import build_answer, unanswered_parts
from us_population_agent.agent.nodes.clarify import build_clarify
from us_population_agent.agent.nodes.execute_sql import SqlExecutor, build_execute_sql
from us_population_agent.agent.nodes.generate_sql import build_generate_sql
from us_population_agent.agent.nodes.intent import build_run_intent
from us_population_agent.agent.nodes.load_columns import ColumnSource, build_load_columns
from us_population_agent.agent.nodes.meta import build_meta
from us_population_agent.agent.nodes.no_data import build_no_data, failed_query, unavailable
from us_population_agent.agent.nodes.off_topic import build_off_topic
from us_population_agent.agent.nodes.repair_sql import build_repair_sql
from us_population_agent.agent.nodes.rewrite import build_rewrite
from us_population_agent.agent.nodes.search import TableSearcher, build_search
from us_population_agent.agent.nodes.select_tables import build_select_tables
from us_population_agent.agent.nodes.validate_sql import build_validate_sql
from us_population_agent.agent.state import AgentState
from us_population_agent.config import Settings, get_settings

Route = Literal["rewrite", "clarify", "meta", "off_topic"]

INTENT_ROUTES: dict[str, Route] = {
    "answerable": "rewrite",
    "needs_clarification": "clarify",
    "meta": "meta",
    "off_topic": "off_topic",
}


def route_after_intent(state: AgentState) -> Route:
    """Picks the next node from the classified intent.

    Args:
        state: Current graph state.

    Returns:
        The node matching the intent. Anything unknown goes to `clarify`.
    """
    return INTENT_ROUTES.get(state.get("intent") or "", "clarify")


def route_after_search(state: AgentState) -> Literal["select_tables", "unavailable"]:
    """Stops with a technical message if every catalog search failed, instead of guessing.

    Args:
        state: Current graph state.

    Returns:
        `unavailable` if all searches failed, otherwise `select_tables`.
    """
    return "unavailable" if state.get("search_failed") else "select_tables"


def route_after_select_tables(state: AgentState) -> Literal["load_columns", "no_table"]:
    """Routes on whether the selector chose a table.

    Args:
        state: Current graph state.

    Returns:
        `load_columns` if a table was selected, otherwise `no_table`.
    """
    return "load_columns" if state.get("selected_table_id") else "no_table"


def route_after_load_columns(state: AgentState) -> Literal["generate_sql", "unavailable"]:
    """Routes on whether the columns of the selected table were loaded.

    Args:
        state: Current graph state.

    Returns:
        `generate_sql` if columns are available, otherwise `unavailable`.
    """
    return "generate_sql" if state.get("table_columns") else "unavailable"


def route_after_generate_sql(state: AgentState) -> Literal["validate_sql", "unavailable", "no_sql"]:
    """Routes on the result of SQL generation (or repair).

    Args:
        state: Current graph state.

    Returns:
        `validate_sql` if a query was produced. Otherwise `unavailable` after a technical failure,
        or `no_sql` if the model decided the data cannot answer the question.
    """
    if state.get("sql"):
        return "validate_sql"
    return "unavailable" if state.get("technical_failure") else "no_sql"


def can_repair(state: AgentState) -> bool:
    """Checks whether the failing query may be rewritten again.

    Args:
        state: Current graph state.

    Returns:
        True if the number of repair attempts is below the `max_repairs` setting.
    """
    return (state.get("sql_attempts") or 0) < get_settings().max_repairs


def route_after_validate_sql(state: AgentState) -> Literal["execute_sql", "repair_sql", "failed_query"]:
    """Routes on the result of SQL validation.

    Args:
        state: Current graph state.

    Returns:
        `execute_sql` if the query passed. Otherwise `repair_sql` while repairs remain, or `failed_query`.
    """
    if not state.get("sql_errors"):
        return "execute_sql"
    return "repair_sql" if can_repair(state) else "failed_query"


def route_after_execute_sql(state: AgentState) -> Literal["answer", "repair_sql", "unavailable", "failed_query"]:
    """Routes on the result of running the query.

    Args:
        state: Current graph state.

    Returns:
        `answer` on success. `unavailable` if Snowflake itself failed, since rewriting the query would not
        help. Otherwise `repair_sql` while repairs remain, or `failed_query`.
    """
    if not state.get("sql_error"):
        return "answer"
    if state.get("technical_failure"):
        return "unavailable"  # Snowflake itself is down; rewriting the query would not help
    return "repair_sql" if can_repair(state) else "failed_query"


def no_table_hint(state: AgentState) -> str | None:
    """Explains why no table fits the question.

    Args:
        state: Current graph state.

    Returns:
        The selector's reason joined with the rewriter's notes (which may say the topic is not in the ACS),
        or None if both are empty.
    """
    return " ".join(part for part in (state.get("selection_reason"), state.get("rewrite_notes")) if part) or None


def no_sql_hint(state: AgentState) -> str | None:
    """Explains why the model wrote no SQL.

    Args:
        state: Current graph state.

    Returns:
        The model's notes joined with the parts of the question it listed as uncovered, or None if there are none.
    """
    parts = [state.get("sql_notes"), *(f"Not covered: {part}." for part in unanswered_parts(state))]
    return " ".join(part for part in parts if part) or None


def build_graph(
    llm: BaseChatModel,
    catalog: TableSearcher,
    columns: ColumnSource,
    runner: SqlExecutor,
    sql_llm: BaseChatModel,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
):
    """Builds the chat graph.

    Conversation history is stored per `thread_id` by the checkpointer.

    Args:
        llm: Light model for classification, rewriting, table selection and replies.
        catalog: Search over the catalog of census tables.
        columns: Source of the columns of a selected table.
        runner: Executor of the generated SQL.
        sql_llm: Strong model that writes and repairs SQL.
        settings: Application settings with the agent limits.
        checkpointer: Checkpointer that persists the state per thread.

    Returns:
        The compiled graph.
    """

    builder = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]  # ty rejects TypedDict states in StateGraph
    builder.add_node("intent", build_run_intent(llm))
    builder.add_node("rewrite", build_rewrite(llm, settings.max_search_queries))
    builder.add_node("search", build_search(catalog, settings.max_candidates))
    builder.add_node("select_tables", build_select_tables(llm))
    builder.add_node("load_columns", build_load_columns(columns))
    builder.add_node("generate_sql", build_generate_sql(sql_llm, settings))
    builder.add_node("repair_sql", build_repair_sql(sql_llm, settings))
    builder.add_node("validate_sql", build_validate_sql(settings))
    builder.add_node("execute_sql", build_execute_sql(runner))
    builder.add_node("answer", build_answer(llm))
    builder.add_node("no_table", build_no_data(llm, name="no_table", hint=no_table_hint))
    builder.add_node("no_sql", build_no_data(llm, name="no_sql", hint=no_sql_hint))
    builder.add_node("failed_query", failed_query)
    builder.add_node("unavailable", unavailable)
    builder.add_node("clarify", build_clarify(llm))
    builder.add_node("meta", build_meta(llm))
    builder.add_node("off_topic", build_off_topic(llm))
    builder.add_edge(START, "intent")
    builder.add_conditional_edges("intent", route_after_intent)
    builder.add_edge("rewrite", "search")
    builder.add_conditional_edges("search", route_after_search)
    builder.add_conditional_edges("select_tables", route_after_select_tables)
    builder.add_conditional_edges("load_columns", route_after_load_columns)
    builder.add_conditional_edges("generate_sql", route_after_generate_sql)
    builder.add_conditional_edges("validate_sql", route_after_validate_sql)
    builder.add_conditional_edges("execute_sql", route_after_execute_sql)
    builder.add_conditional_edges("repair_sql", route_after_generate_sql)
    builder.add_edge("answer", END)
    builder.add_edge("no_table", END)
    builder.add_edge("no_sql", END)
    builder.add_edge("failed_query", END)
    builder.add_edge("unavailable", END)
    builder.add_edge("clarify", END)
    builder.add_edge("meta", END)
    builder.add_edge("off_topic", END)
    return builder.compile(checkpointer=checkpointer)
