from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the US population. "
    "Be concise. Never invent numbers; if you lack data, say so clearly."
)


def build_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver):
    """Build the chat graph. Conversation history is stored per `thread_id` by the checkpointer."""

    def call_model(state: MessagesState) -> dict:
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": response}

    builder = StateGraph(MessagesState)  # ty: ignore[invalid-argument-type]  # ty does not recognise MessagesState as TypedDict
    builder.add_node("chat", call_model)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)
