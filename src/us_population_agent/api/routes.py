import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from us_population_agent.api.dependencies import get_graph, get_settings, require_api_key
from us_population_agent.api.schemas import ChatRequest, ChatResponse
from us_population_agent.api.service import TurnFailed, TurnTimeout, run_turn
from us_population_agent.config import Settings

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Reports that the service is running.

    Returns:
        A dict with `status` set to `ok`.
    """
    return {"status": "ok"}


@router.post("/chat", dependencies=[Depends(require_api_key)])
async def chat(
    body: ChatRequest,
    graph: Annotated[object, Depends(get_graph)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    """Answers one question, continuing the conversation identified by `thread_id`.

    A new thread is started if the request has no `thread_id`.

    Args:
        body: The question and an optional thread id.
        graph: The agent graph.
        settings: Application settings with the message length and turn timeout limits.

    Returns:
        The answer together with the id of the thread it belongs to.

    Raises:
        HTTPException: With status 422 if the message is too long, 504 if the question takes longer
            than the turn timeout, and 500 if the turn fails.
    """
    if len(body.message) > settings.max_message_chars:
        raise HTTPException(status_code=422, detail=f"Message longer than {settings.max_message_chars} characters")

    thread_id = body.thread_id or uuid.uuid4().hex
    try:
        answer = await run_turn(graph, thread_id, body.message, settings.turn_timeout_seconds)
    except TurnTimeout as exc:
        raise HTTPException(status_code=504, detail="The question took too long, please try again") from exc
    except TurnFailed as exc:
        raise HTTPException(status_code=500, detail="Something went wrong, please try again") from exc
    return ChatResponse(thread_id=thread_id, answer=answer)
