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
    """Builds the Anthropic chat model configured from the application settings.

    Args:
        model_name: Model to use. Defaults to `anthropic_model_name` from the settings.
        effort: Reasoning effort level passed to the model. If None, the model's default is used.

    Returns:
        A chat model with the configured API key, retry count and timeout.

    Raises:
        LLMInitError: If the settings are invalid or the model cannot be created.
    """
    try:
        settings = get_settings()
        return ChatAnthropic(
            model=model_name or settings.anthropic_model_name,
            effort=effort,
            api_key=settings.anthropic_api_key,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_seconds,
        )
    except Exception as e:
        logger.exception("LLM initialisation failed")
        raise LLMInitError("Cannot initialise the LLM, check ANTHROPIC_API_KEY and model name") from e
