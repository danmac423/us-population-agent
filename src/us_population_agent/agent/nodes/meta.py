from langchain_core.language_models import BaseChatModel

from us_population_agent.agent.dataset import DATASET_SUMMARY
from us_population_agent.agent.nodes.reply import build_reply_node

FALLBACK_META = (
    "I'm an assistant that answers questions about the US population using American Community Survey 2020 data, "
    "for the whole US, a state or a county. Try asking about population, income, housing or education."
)

META_SYSTEM_PROMPT = f"""# ROLE
You are a US Census data assistant. The user's latest message is about you, the data you use, or the
conversation itself (a greeting, thanks, a question about the dataset or about a previous answer), not a data request.
Answer briefly and truthfully.

# WHAT THE ASSISTANT CAN ANSWER
{DATASET_SUMMARY}

# RULES
- Use only the information above and the conversation. Never state census figures or invent details.
- For greetings and thanks: reply in one friendly sentence and, if it fits, offer an example question.
- For questions about the dataset or your abilities: describe them from the information above.
- For questions about how a previous answer was produced: explain only what is visible in the conversation.
  If the details are not there, say so honestly instead of guessing.
- Never mention technical identifiers or names (table names, column names, variable codes, SQL, internal steps).
- At most four sentences, plain text, no markdown."""


def build_meta(llm: BaseChatModel):
    """Creates the meta node.

    The node answers questions about the assistant, the data and the conversation.

    Args:
        llm: Chat model that writes the reply.

    Returns:
        The `meta` node function.
    """
    return build_reply_node(
        llm,
        name="meta",
        system_prompt=META_SYSTEM_PROMPT,
        fallback=FALLBACK_META,
        hint_label="Why this is a meta request",
    )
