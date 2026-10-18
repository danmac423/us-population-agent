from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from us_population_agent.api import create_app
from us_population_agent.config import Settings


class FakeGraph:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    async def ainvoke(self, state, config):
        self.calls.append((state, config))
        if self.error:
            raise self.error
        return {"messages": [AIMessage("42")]}


def make_client(graph: FakeGraph, **overrides) -> TestClient:
    settings = Settings.model_construct(log_level="WARNING", **overrides)

    @asynccontextmanager
    async def factory(_settings):
        yield graph

    return TestClient(create_app(settings, factory))


def test_health():
    with make_client(FakeGraph()) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_chat_starts_a_new_thread_when_none_is_given():
    graph = FakeGraph()
    with make_client(graph) as client:
        response = client.post("/chat", json={"message": "hi"})

    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "42"
    assert body["thread_id"] == graph.calls[0][1]["configurable"]["thread_id"]


def test_chat_reuses_the_given_thread():
    graph = FakeGraph()
    with make_client(graph) as client:
        response = client.post("/chat", json={"message": "hi", "thread_id": "abc-1"})

    assert response.json()["thread_id"] == "abc-1"


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "hi", "thread_id": "bad id!"}, {}])
def test_invalid_requests_are_rejected(body):
    with make_client(FakeGraph()) as client:
        assert client.post("/chat", json=body).status_code == 422


def test_too_long_message_is_rejected():
    graph = FakeGraph()
    with make_client(graph, max_message_chars=5, turn_timeout_seconds=5) as client:
        response = client.post("/chat", json={"message": "x" * 6})

    assert response.status_code == 422
    assert graph.calls == []


def test_graph_failure_gives_500_without_details():
    with make_client(FakeGraph(RuntimeError("secret detail")), turn_timeout_seconds=5) as client:
        response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 500
    assert "secret" not in response.text


def test_api_key_is_enforced_when_configured():
    with make_client(FakeGraph(), api_key=SecretStr("k"), turn_timeout_seconds=5, max_message_chars=100) as client:
        assert client.post("/chat", json={"message": "hi"}).status_code == 401
        assert client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "nope"}).status_code == 401
        assert client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "k"}).status_code == 200
