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
    """Create a node that answers with a single LLM-written message and appends it to the history.

    The hint (by default the classifier's reason) is passed to the model. When the model fails or returns nothing, `fallback`
    is used, so the user never gets an empty answer.
    """

    def reply(state: AgentState) -> dict:
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
