from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from us_population_agent.agent.dataset import DATASET_SUMMARY
from us_population_agent.agent.nodes.reply import build_reply_node
from us_population_agent.agent.state import AgentState

# Used when the model fails or returns nothing: the user must never get an empty answer.
FALLBACK_NO_DATA = (
    "I couldn't answer that from the census data I have. Could you rephrase the question, "
    "or ask about a different measure (for example population, income, housing or poverty) or place?"
)
UNAVAILABLE = "Something went wrong while looking up the data. Please try again in a moment."
FAILED_QUERY = (
    "I couldn't build a reliable query for that question. "
    "Could you rephrase it, or ask about a narrower measure or place?"
)

NO_DATA_SYSTEM_PROMPT = f"""# ROLE
You are a US Census data assistant. The user's question is reasonable but cannot be answered from the data you
have. Explain this clearly and helpfully, without guessing any number.

# WHAT THE ASSISTANT CAN ANSWER
{DATASET_SUMMARY}

# RULES
- Say plainly what could not be answered and why, using the "Why it cannot be answered" hint. Do not quote the
  hint and do not mention tables, columns, SQL or internal steps.
- If the hint says part of the question can be answered, say which part, and offer to answer just that part.
- Suggest one or two concrete alternatives that the data really supports (a related measure, a state or county
  instead of a city, a different breakdown).
- Never state or estimate figures and never claim you looked something up.
- Be brief and polite: at most three sentences, plain text, no markdown."""


def build_no_data(llm: BaseChatModel, *, name: str, hint: Callable[[AgentState], str | None]):
    """Creates a node that tells the user the question cannot be answered.

    The stage's own reason is passed to the model as a hint.

    Args:
        llm: Chat model that writes the reply.
        name: Name of the node, used in logs.
        hint: Function that returns why the question cannot be answered, or None if there is no reason.

    Returns:
        The node function.
    """
    return build_reply_node(
        llm,
        name=name,
        system_prompt=NO_DATA_SYSTEM_PROMPT,
        fallback=FALLBACK_NO_DATA,
        hint_label="Why it cannot be answered",
        hint=hint,
    )


def unavailable(state: AgentState) -> dict:
    """Replies with a fixed message after a technical failure, without calling the model.

    Covers the catalog, metadata, Snowflake or the model being unavailable.

    Args:
        state: Current graph state (unused).

    Returns:
        A state update with the message appended to `messages`.
    """
    return {"messages": [AIMessage(UNAVAILABLE)]}


def failed_query(state: AgentState) -> dict:
    """Replies with a fixed message when the generated query kept failing after repairs, without calling the model.

    Args:
        state: Current graph state (unused).

    Returns:
        A state update with the message appended to `messages`.
    """
    return {"messages": [AIMessage(FAILED_QUERY)]}
