from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def open_checkpointer(database_url: str) -> AsyncGenerator[AsyncPostgresSaver]:
    """Opens the async Postgres checkpointer and creates its tables if needed.

    The graph using this checkpointer must be run through its async API (`ainvoke`).

    Args:
        database_url: Postgres connection string.

    Yields:
        A ready-to-use checkpointer. The connection is closed when the context exits.
    """
    async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
