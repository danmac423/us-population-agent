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
    """Build the models and open Snowflake and Postgres; yield the ready graph and close everything on exit.

    Shared by the CLI and the web API. Raises `LLMInitError`, `SnowflakeInitError` or `psycopg.OperationalError`
    when a resource is not available.
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
