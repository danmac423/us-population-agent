import logging
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from us_population_agent.agent.context import format_context
from us_population_agent.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_reply_node(
    llm: BaseChatModel,
    *,
    name: str,
    system_prompt: str,
    fallback: str,
    hint_label: str,
    hint: Callable[[AgentState], str | None] | None = None,
):
    """Creates a node that answers with a single LLM-written message and appends it to the history.

    The hint is passed to the model. When the model fails or returns
    nothing, `fallback` is used, so the user never gets an empty answer.

    Args:
        llm: Chat model that writes the reply.
        name: Name of the node, used in logs.
        system_prompt: System prompt for the model.
        fallback: Message used when the model fails or returns nothing.
        hint_label: Label under which the hint is shown to the model.
        hint: Function that returns the hint for the current state. Defaults to the intent classifier's reason.

    Returns:
        The `reply` node function.
    """

    def reply(state: AgentState) -> dict:
        """Writes one reply to the latest user message.

        If the latest message is not from the user, returns `fallback` without calling the model.

        Args:
            state: Current graph state.

        Returns:
            A state update with the reply appended to `messages`.
        """
        *history, latest = state["messages"]
        if not isinstance(latest, HumanMessage):
            logger.warning("%s: latest message is not from the user (%s)", name, latest.type)
            return {"messages": [AIMessage(fallback)]}

        prompt = (
            f"Conversation so far:\n{format_context(history)}\n\n"
            f"Latest user message:\n{latest.text}\n\n"
            f"{hint_label}:\n{(hint(state) if hint else state.get('intent_reason')) or 'Not specified.'}"
        )
        try:
            text = llm.invoke([SystemMessage(system_prompt), HumanMessage(prompt)]).text.strip()
        except Exception:
            logger.exception("%s: generating the reply failed", name)
            text = ""

        return {"messages": [AIMessage(text or fallback)]}

    return reply
