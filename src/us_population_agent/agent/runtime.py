import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from us_population_agent.agent.graph import build_graph
from us_population_agent.config import Settings
from us_population_agent.llm.factory import build_llm
from us_population_agent.persistence.checkpointer import open_checkpointer
from us_population_agent.snowflake_client.columns import ColumnCatalog
from us_population_agent.snowflake_client.connection import open_connection
from us_population_agent.snowflake_client.query import QueryRunner
from us_population_agent.snowflake_client.search import CatalogSearch

logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_graph(settings: Settings) -> AsyncGenerator:
    """Builds the models, opens Snowflake and Postgres, and yields the ready graph.

    Args:
        settings: Application settings for the models, Snowflake and the Postgres checkpointer.

    Yields:
        The compiled graph, ready to be run through its async API.

    Raises:
        LLMInitError: If a chat model cannot be created.
        SnowflakeInitError: If the Snowflake connection cannot be opened.
        psycopg.OperationalError: If the Postgres database is not available.
    """
    llm = build_llm(model_name=settings.anthropic_model_name, effort="low")
    sql_llm = build_llm(model_name=settings.anthropic_strong_model_name, effort="low")
    logger.info("LLM ready")

    with open_connection(settings) as connection:
        async with open_checkpointer(settings.database_url) as checkpointer:
            graph = build_graph(
                llm,
                CatalogSearch(connection, settings),
                ColumnCatalog(connection, settings),
                QueryRunner(connection, settings),
                sql_llm,
                settings,
                checkpointer,
            )
            yield graph
