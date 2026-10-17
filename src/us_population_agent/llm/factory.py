import logging
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from us_population_agent.config import get_settings

logger = logging.getLogger(__name__)


class LLMInitError(Exception):
    """Raised when the chat model cannot be created."""


def build_llm(
    model_name: str | None = None, effort: Literal["max", "xhigh", "high", "medium", "low"] | None = None
) -> BaseChatModel:
    try:
        settings = get_settings()
        return ChatAnthropic(
            model=model_name or settings.anthropic_model_name,
            effort=effort,
            api_key=settings.anthropic_api_key,
            timeout=30,
        )
    except Exception as e:
        logger.exception("LLM initialisation failed")
        raise LLMInitError("Cannot initialise the LLM, check ANTHROPIC_API_KEY and model name") from e
