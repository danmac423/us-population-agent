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
UNAVAILABLE = "I couldn't reach the census data just now. Please try again in a moment."

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
    """Create a node that tells the user the question cannot be answered, using the stage's own reason as the hint."""
    return build_reply_node(
        llm,
        name=name,
        system_prompt=NO_DATA_SYSTEM_PROMPT,
        fallback=FALLBACK_NO_DATA,
        hint_label="Why it cannot be answered",
        hint=hint,
    )


def unavailable(state: AgentState) -> dict:
    """Technical failure (catalog or metadata unreachable): a fixed message, no model call."""
    return {"messages": [AIMessage(UNAVAILABLE)]}
