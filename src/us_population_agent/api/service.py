import asyncio
import logging

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class TurnTimeout(Exception):
    """The agent did not answer within the time limit."""


class TurnFailed(Exception):
    """The agent raised an unexpected error."""


async def run_turn(graph, thread_id: str, message: str, timeout_seconds: int) -> str:
    """Send one user message to the agent in the given conversation and return its reply."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        async with asyncio.timeout(timeout_seconds):
            result = await graph.ainvoke({"messages": [HumanMessage(message)]}, config)
    except TimeoutError as exc:
        logger.warning("Turn timed out (thread %s)", thread_id)
        raise TurnTimeout from exc
    except Exception as exc:
        logger.exception("The turn failed (thread %s)", thread_id)
        raise TurnFailed from exc
    return str(result["messages"][-1].content)
