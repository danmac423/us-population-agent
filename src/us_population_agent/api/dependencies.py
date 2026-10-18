import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from us_population_agent.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_graph(request: Request):
    return request.app.state.graph


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)], x_api_key: Annotated[str | None, Header()] = None
) -> None:
    """Reject the request unless it carries the configured API key; no key configured means open access."""
    if settings.api_key is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key.get_secret_value()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
