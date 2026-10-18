from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from us_population_agent.agent.schemas import Intent


class AgentState(TypedDict):
    """Graph state, persisted per thread by the checkpointer.

    Keep it to raw data that later nodes need. Prompts, formatted context and clients stay local to the nodes.
    The remaining fields are per-turn results, overwritten by their node on every question. They are kept as
    plain values because custom types in checkpoints are going to be blocked by LangGraph.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    # intent node
    intent: Intent | None
    intent_reason: str | None
    # rewrite node
    queries: list[str] | None
    rewrite_notes: str | None
    # search node
    candidates: list[dict] | None
    search_failed: bool | None
    # select_tables node
    selected_table_id: str | None
    selection_reason: str | None
    # load_columns node
    data_table_name: str | None
    table_prefix: str | None  # table, topic, title and universe shared by all columns
    table_columns: list[dict] | None
    # generate_sql node
    sql: str | None
    sql_notes: str | None
    sql_columns: list[dict] | None
    sql_attempts: int | None  # how many times repair_sql has rewritten the query
    # validate_sql node
    sql_errors: list[str] | None  # problems found in `sql`; empty when it passed
    # execute_sql node
    sql_result: dict | None  # columns, rows
    sql_error: str | None
