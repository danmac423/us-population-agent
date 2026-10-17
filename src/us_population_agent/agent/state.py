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
