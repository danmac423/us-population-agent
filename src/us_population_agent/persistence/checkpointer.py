from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def open_checkpointer(database_url: str) -> AsyncGenerator[AsyncPostgresSaver]:
    """Open the async Postgres checkpointer; the graph must be used through its async API (`ainvoke`)."""
    async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
