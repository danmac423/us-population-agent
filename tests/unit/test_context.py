from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from us_population_agent.agent.context import format_context


def test_empty_history():
    assert format_context([], max_tokens=1000) == "(none)"


def test_short_history_is_kept_whole_and_tool_messages_are_skipped():
    messages = [
        HumanMessage("How many people live in Texas?"),
        ToolMessage("raw", tool_call_id="1"),
        AIMessage("29 million"),
    ]

    assert format_context(messages, max_tokens=1000) == "User: How many people live in Texas?\nAssistant: 29 million"


def test_oldest_messages_are_dropped_when_history_is_too_long():
    messages = []
    for i in range(50):
        messages += [HumanMessage(f"question {i} " + "x" * 200), AIMessage(f"answer {i} " + "y" * 200)]

    text = format_context(messages, max_tokens=300)

    assert "question 49" in text
    assert "answer 49" in text
    assert "question 0 " not in text
    assert text.startswith("User: ")  # never opens with an orphaned assistant reply
