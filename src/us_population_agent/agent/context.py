from collections.abc import Sequence

from langchain_core.messages import AnyMessage, filter_messages, get_buffer_string, trim_messages

from us_population_agent.config import get_settings


def format_context(messages: Sequence[AnyMessage], max_tokens: int | None = None) -> str:
    """Render the latest part of the conversation as a plain transcript (tool messages are skipped).

    The oldest messages are dropped until the rest fits in `max_tokens` (estimated; default: the
    `max_context_tokens` setting), and the transcript always
    starts with a user message so it never opens in the middle of an exchange.
    """
    max_tokens = max_tokens or get_settings().max_context_tokens
    turns = filter_messages(messages, include_types=["human", "ai"])
    recent = trim_messages(turns, max_tokens=max_tokens, token_counter="approximate", strategy="last", start_on="human")
    return get_buffer_string(recent, human_prefix="User", ai_prefix="Assistant") or "(none)"
