from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from us_population_agent.agent.nodes.clarify import build_clarify
from us_population_agent.agent.nodes.intent import build_run_intent
from us_population_agent.agent.nodes.meta import build_meta
from us_population_agent.agent.nodes.off_topic import build_off_topic
from us_population_agent.agent.nodes.rewrite import build_rewrite
from us_population_agent.agent.nodes.search import TableSearcher, build_search
from us_population_agent.agent.nodes.select_tables import build_select_tables
from us_population_agent.agent.state import AgentState

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the US population. "
    "Be concise. Never invent numbers; if you lack data, say so clearly."
)


Route = Literal["rewrite", "clarify", "meta", "off_topic", "chat"]

INTENT_ROUTES: dict[str, Route] = {
    "answerable": "rewrite",
    "needs_clarification": "clarify",
    "meta": "meta",
    "off_topic": "off_topic",
}


def route_after_intent(state: AgentState) -> Route:
    """Pick the next node from the classified intent. Anything unknown goes to `chat`."""
    return INTENT_ROUTES.get(state.get("intent") or "", "chat")


def build_graph(
    llm: BaseChatModel, intent_llm: BaseChatModel, catalog: TableSearcher, checkpointer: BaseCheckpointSaver
):
    """Build the chat graph. Conversation history is stored per `thread_id` by the checkpointer."""

    def call_model(state: AgentState) -> dict:
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": response}

    builder = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]  # ty rejects TypedDict states in StateGraph
    builder.add_node("intent", build_run_intent(intent_llm))
    builder.add_node("rewrite", build_rewrite(intent_llm))
    builder.add_node("search", build_search(catalog))
    builder.add_node("select_tables", build_select_tables(llm))
    builder.add_node("clarify", build_clarify(llm))
    builder.add_node("meta", build_meta(llm))
    builder.add_node("off_topic", build_off_topic(llm))
    builder.add_node("chat", call_model)
    builder.add_edge(START, "intent")
    builder.add_conditional_edges("intent", route_after_intent)
    builder.add_edge("rewrite", "search")
    builder.add_edge("search", "select_tables")
    builder.add_edge("select_tables", "chat")  # temporary: columns and SQL come next
    builder.add_edge("clarify", END)
    builder.add_edge("meta", END)
    builder.add_edge("off_topic", END)
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)
