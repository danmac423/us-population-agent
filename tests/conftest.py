from types import SimpleNamespace
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage

from us_population_agent.agent.state import AgentState


def state(**fields) -> AgentState:
    return cast(AgentState, fields)


def question_state(text: str = "How many people live in Texas?", **fields) -> AgentState:
    return state(messages=[HumanMessage(text)], **fields)


class FakeStructuredLLM:
    """Stands in for a chat model: `with_structured_output` and `invoke` return the prepared outcome.

    `outcome` is returned as is (a string becomes an AIMessage) or raised when it is an exception.
    Prompts of every call are kept in `calls`.
    """

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls: list[list] = []

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if isinstance(self.outcome, str):
            return AIMessage(self.outcome)
        return self.outcome

    @property
    def last_prompt(self) -> str:
        return self.calls[-1][-1].content


class FakeCursor:
    def __init__(self, rows=None, description=None, error: Exception | None = None):
        self.rows = rows or []
        self.description = description
        self.error = error
        self.executed: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None, **kwargs):
        self.executed.append((sql, params, kwargs))
        if self.error:
            raise self.error

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def column(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)
