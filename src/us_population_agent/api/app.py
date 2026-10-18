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
    """Creates the FastAPI app.

    The graph is built once at startup and shared by all requests.

    Args:
        settings: Application settings. Defaults to the settings read from the environment.
        graph_factory: Factory that opens the graph for the given settings. Replaceable in tests.

    Returns:
        The configured app.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        """Sets up logging and the graph at startup and closes them at shutdown.

        Args:
            app: The app whose state receives `settings` and `graph`.

        Yields:
            Control to the app while it serves requests.
        """
        setup_logging(settings.log_level)
        async with graph_factory(settings) as graph:
            app.state.settings = settings
            app.state.graph = graph
            yield

    app = FastAPI(title="US population assistant", lifespan=lifespan)
    app.include_router(router)
    return app
