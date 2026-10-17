from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from us_population_agent.agent.nodes.answer import build_answer
from us_population_agent.agent.nodes.clarify import build_clarify
from us_population_agent.agent.nodes.execute_sql import SqlExecutor, build_execute_sql
from us_population_agent.agent.nodes.generate_sql import build_generate_sql
from us_population_agent.agent.nodes.intent import build_run_intent
from us_population_agent.agent.nodes.load_columns import ColumnSource, build_load_columns
from us_population_agent.agent.nodes.meta import build_meta
from us_population_agent.agent.nodes.off_topic import build_off_topic
from us_population_agent.agent.nodes.rewrite import build_rewrite
from us_population_agent.agent.nodes.search import TableSearcher, build_search
from us_population_agent.agent.nodes.select_tables import build_select_tables
from us_population_agent.agent.state import AgentState
from us_population_agent.config import Settings

Route = Literal["rewrite", "clarify", "meta", "off_topic"]

INTENT_ROUTES: dict[str, Route] = {
    "answerable": "rewrite",
    "needs_clarification": "clarify",
    "meta": "meta",
    "off_topic": "off_topic",
}


def route_after_intent(state: AgentState) -> Route:
    """Pick the next node from the classified intent. Anything unknown goes to `clarify`."""
    return INTENT_ROUTES.get(state.get("intent") or "", "clarify")


def build_graph(
    llm: BaseChatModel,
    intent_llm: BaseChatModel,
    catalog: TableSearcher,
    columns: ColumnSource,
    runner: SqlExecutor,
    sql_llm: BaseChatModel,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
):
    """Build the chat graph. Conversation history is stored per `thread_id` by the checkpointer."""

    builder = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]  # ty rejects TypedDict states in StateGraph
    builder.add_node("intent", build_run_intent(intent_llm))
    builder.add_node("rewrite", build_rewrite(intent_llm))
    builder.add_node("search", build_search(catalog))
    builder.add_node("select_tables", build_select_tables(llm))
    builder.add_node("load_columns", build_load_columns(columns))
    builder.add_node("generate_sql", build_generate_sql(sql_llm, settings))
    builder.add_node("execute_sql", build_execute_sql(runner))
    builder.add_node("answer", build_answer(llm))
    builder.add_node("clarify", build_clarify(llm))
    builder.add_node("meta", build_meta(llm))
    builder.add_node("off_topic", build_off_topic(llm))
    builder.add_edge(START, "intent")
    builder.add_conditional_edges("intent", route_after_intent)
    builder.add_edge("rewrite", "search")
    builder.add_edge("search", "select_tables")
    builder.add_edge("select_tables", "load_columns")
    builder.add_edge("load_columns", "generate_sql")
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_edge("execute_sql", "answer")
    builder.add_edge("answer", END)
    builder.add_edge("clarify", END)
    builder.add_edge("meta", END)
    builder.add_edge("off_topic", END)
    return builder.compile(checkpointer=checkpointer)
