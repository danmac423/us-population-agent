import logging
import sys

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from us_population_agent.config import get_settings
from us_population_agent.llm.factory import LLMInitError, build_llm
from us_population_agent.logging_config import setup_logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the US population. "
    "Be concise. Never invent numbers; if you lack data, say so clearly."
)
EXIT_COMMANDS = {"exit", "quit"}


def main() -> None:
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

    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    print("US population assistant. Type 'exit' to quit.")

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

        messages.append(HumanMessage(text))
        response = llm.invoke(messages)
        messages.append(response)
        print(response.content)
