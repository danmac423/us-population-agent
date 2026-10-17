import argparse
import logging
import sys

import psycopg
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from us_population_agent.agent.graph import build_graph
from us_population_agent.config import get_settings
from us_population_agent.llm.factory import LLMInitError, build_llm
from us_population_agent.logging_config import setup_logging
from us_population_agent.persistance.checkpointer import open_checkpointer

logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"exit", "quit"}
DEFAULT_THREAD_ID = "demo"


def main() -> None:
    parser = argparse.ArgumentParser(description="US population assistant")
    parser.add_argument("--thread", default=DEFAULT_THREAD_ID, help="conversation id; reuse it to resume a chat")
    args = parser.parse_args()

    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Invalid configuration (check .env):\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    setup_logging(settings.log_level)

    try:
        llm = build_llm(effort="low")
    except LLMInitError as exc:
        logger.error("Startup error: %s", exc)
        raise SystemExit(1) from exc
    logger.info("LLM ready")

    try:
        with open_checkpointer(settings.database_url) as checkpointer:
            graph = build_graph(llm, checkpointer)
            run_chat(graph, args.thread)
    except psycopg.OperationalError as exc:
        logger.error("Database error, check DATABASE_URL and that the database is running: %s", exc)
        raise SystemExit(1) from exc


def run_chat(graph, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"US population assistant (conversation: {thread_id}). Type 'exit' to quit.")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in EXIT_COMMANDS:
            break

        result = graph.invoke({"messages": [HumanMessage(text)]}, config)
        print(result["messages"][-1].content)
