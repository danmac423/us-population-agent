import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from us_population_agent.config import Settings


def get_settings(request: Request) -> Settings:
    """FastAPI dependency that provides the application settings.

    Args:
        request: The incoming request.

    Returns:
        The settings stored in the app state at startup.
    """
    return request.app.state.settings


def get_graph(request: Request):
    """FastAPI dependency that provides the agent graph.

    Args:
        request: The incoming request.

    Returns:
        The graph stored in the app state at startup.
    """
    return request.app.state.graph


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)], x_api_key: Annotated[str | None, Header()] = None
) -> None:
    """FastAPI dependency that rejects the request unless it carries the configured API key.

    If no key is configured, access is open.

    Args:
        settings: Application settings holding the expected API key.
        x_api_key: Value of the `X-API-Key` header.

    Raises:
        HTTPException: With status 401 if the key is missing or does not match.
    """
    if settings.api_key is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key.get_secret_value()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
