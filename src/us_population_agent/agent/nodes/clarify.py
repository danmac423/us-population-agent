from langchain_core.language_models import BaseChatModel

from us_population_agent.agent.dataset import DATASET_SUMMARY
from us_population_agent.agent.nodes.reply import build_reply_node

# Used when the model fails or returns nothing: the user must never get an empty answer.
FALLBACK_QUESTION = (
    "I need a bit more detail to answer that. Could you tell me which measure "
    "(for example population, income or housing) and which state or county you mean?"
)

CLARIFY_SYSTEM_PROMPT = f"""# ROLE
You are a US Census data assistant. The user's latest message cannot be answered yet because
something is missing or ambiguous. Ask the user ONE short clarifying question.

# WHAT THE ASSISTANT CAN ANSWER
{DATASET_SUMMARY}

# RULES
- Ask exactly one question. Do not answer the original question and do not state any numbers.
- Use the "Why it is unclear" hint to decide what to ask, but do not quote or mention it.
- When it helps, offer 2-4 concrete options the assistant really supports.
- Be polite and brief: at most two sentences, plain text, no markdown.

# EXAMPLES
Why it is unclear: Geography is clear (FL) but no measure was given.
Question: Which topic about Florida would you like to see, for example population, income or housing?

Why it is unclear: 'Seattle' is a city, which is not a supported geography.
Question: I can't look up cities directly. Would you like King County, WA or the state of Washington instead?"""


def build_clarify(llm: BaseChatModel):
    """Create the clarify node. It asks the user for the missing detail and adds the question to the history."""
    return build_reply_node(
        llm,
        name="clarify",
        system_prompt=CLARIFY_SYSTEM_PROMPT,
        fallback=FALLBACK_QUESTION,
        hint_label="Why it is unclear",
    )
