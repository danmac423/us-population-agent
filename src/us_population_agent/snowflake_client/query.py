from dataclasses import dataclass
from decimal import Decimal

import snowflake.connector
from snowflake.connector import SnowflakeConnection

from us_population_agent.config import Settings


class QueryError(Exception):
    """Raised when Snowflake rejects a generated query (bad SQL), so rewriting the query can help."""


class QueryUnavailableError(QueryError):
    """Raised when Snowflake cannot run queries at all (connection, network, service), so rewriting will not help."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list]  # plain values only (see `to_plain`)


def to_plain(value):
    """Convert a Snowflake value to a plain one that the checkpointer can store."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return str(value)


class QueryRunner:
    """Runs generated SELECT queries on the census data (the connection's role is read-only)."""

    def __init__(self, connection: SnowflakeConnection, settings: Settings) -> None:
        self._connection = connection
        self._timeout_seconds = settings.query_timeout_seconds

    def run(self, sql: str) -> QueryResult:
        """Execute the query and return all rows. Raises `QueryError`, or `QueryUnavailableError` for infrastructure errors."""
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(sql, timeout=self._timeout_seconds)
                rows = cursor.fetchall()
                columns = [column.name for column in cursor.description or []]
        except snowflake.connector.ProgrammingError as exc:
            raise QueryError(str(exc)) from exc
        except snowflake.connector.Error as exc:
            raise QueryUnavailableError(str(exc)) from exc

        return QueryResult(columns=columns, rows=[[to_plain(value) for value in row] for row in rows])
