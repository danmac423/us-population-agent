import logging
from collections.abc import Generator
from contextlib import contextmanager

import snowflake.connector
from snowflake.connector import SnowflakeConnection

from us_population_agent.config import Settings

logger = logging.getLogger(__name__)


class SnowflakeInitError(Exception):
    """Raised when the Snowflake connection cannot be opened."""


@contextmanager
def open_connection(settings: Settings) -> Generator[SnowflakeConnection]:
    """Open a Snowflake connection that lives for the duration of the `with` block."""
    try:
        connection = snowflake.connector.connect(
            account=settings.snowflake_account,
            user=settings.snowflake_user,
            password=settings.snowflake_password.get_secret_value(),
            role=settings.snowflake_role,
            warehouse=settings.snowflake_warehouse,
            database=settings.snowflake_database,
            schema=settings.snowflake_schema,
            login_timeout=settings.login_timeout_seconds,
        )
    except snowflake.connector.Error as exc:
        logger.exception("Cannot connect to Snowflake")
        raise SnowflakeInitError(
            "Cannot connect to Snowflake, check the SNOWFLAKE_* settings and the token or password"
        ) from exc

    try:
        yield connection
    finally:
        connection.close()
