from collections.abc import Generator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver


@contextmanager
def open_checkpointer(database_url: str) -> Generator[PostgresSaver]:
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()
        yield checkpointer
