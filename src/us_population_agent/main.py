import argparse
import asyncio
import logging
import sys

import psycopg
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from us_population_agent.agent.graph import build_graph
from us_population_agent.config import Settings, get_settings
from us_population_agent.llm.factory import LLMInitError, build_llm
from us_population_agent.logging_config import setup_logging
from us_population_agent.persistence.checkpointer import open_checkpointer
from us_population_agent.snowflake_client.columns import ColumnCatalog
from us_population_agent.snowflake_client.connection import SnowflakeInitError, open_connection
from us_population_agent.snowflake_client.query import QueryRunner
from us_population_agent.snowflake_client.search import CatalogSearch

logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"exit", "quit"}
TURN_FAILED = "Something went wrong while handling that. Please try again."


def main() -> None:
    parser = argparse.ArgumentParser(description="US population assistant")
    parser.add_argument("--thread", help="conversation id; reuse it to resume a chat (default: DEFAULT_THREAD_ID)")
    args = parser.parse_args()

    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Invalid configuration (check .env):\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    setup_logging(settings.log_level)

    try:
        llm = build_llm(model_name=settings.anthropic_model_name, effort="low")
        sql_llm = build_llm(model_name=settings.anthropic_strong_model_name, effort="low")
    except LLMInitError as exc:
        logger.error("Startup error: %s", exc)
        raise SystemExit(1) from exc
    logger.info("LLM ready")

    try:
        asyncio.run(chat(settings, llm, sql_llm, args.thread or settings.default_thread_id))
    except SnowflakeInitError as exc:
        logger.error("Startup error: %s", exc)
        raise SystemExit(1) from exc
    except psycopg.OperationalError as exc:
        logger.error("Database error, check DATABASE_URL and that the database is running: %s", exc)
        raise SystemExit(1) from exc


async def chat(settings: Settings, llm, sql_llm, thread_id: str) -> None:
    with open_connection(settings) as connection:
        async with open_checkpointer(settings.database_url) as checkpointer:
            catalog = CatalogSearch(connection, settings)
            columns = ColumnCatalog(connection, settings)
            runner = QueryRunner(connection, settings)
            graph = build_graph(llm, catalog, columns, runner, sql_llm, settings, checkpointer)
            await run_chat(graph, thread_id)


async def run_chat(graph, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"US population assistant (conversation: {thread_id}). Type 'exit' to quit.")

    while True:
        try:
            text = (await asyncio.to_thread(input, "> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in EXIT_COMMANDS:
            break

        try:
            result = await graph.ainvoke({"messages": [HumanMessage(text)]}, config)
        except Exception:
            # A failure inside one turn must not end the whole conversation.
            logger.exception("The turn failed")
            print(TURN_FAILED)
            continue
        print(result["messages"][-1].content)
