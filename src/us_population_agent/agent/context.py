from collections.abc import Sequence

from langchain_core.messages import AnyMessage, filter_messages, get_buffer_string


def format_context(messages: Sequence[AnyMessage]) -> str:
    """Render the whole conversation as a plain transcript (tool messages are skipped). No trimming yet."""
    turns = filter_messages(messages, include_types=["human", "ai"])
    return get_buffer_string(turns, human_prefix="User", ai_prefix="Assistant") or "(none)"
