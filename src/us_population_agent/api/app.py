from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from us_population_agent.agent.runtime import open_graph
from us_population_agent.api.routes import router
from us_population_agent.config import Settings, get_settings
from us_population_agent.logging_config import setup_logging


def create_app(
    settings: Settings | None = None,
    graph_factory: Callable[[Settings], AbstractAsyncContextManager] = open_graph,
) -> FastAPI:
    """Create the app. The graph is built once at startup and shared by all requests."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        setup_logging(settings.log_level)
        async with graph_factory(settings) as graph:
            app.state.settings = settings
            app.state.graph = graph
            yield

    app = FastAPI(title="US population assistant", lifespan=lifespan)
    app.include_router(router)
    return app
