from dataclasses import dataclass
from decimal import Decimal

import snowflake.connector
from snowflake.connector import SnowflakeConnection

MAX_ROWS = 100
TIMEOUT_SECONDS = 30


class QueryError(Exception):
    """Raised when a generated query fails in Snowflake."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list]  # plain values only (see `to_plain`), at most MAX_ROWS
    truncated: bool  # True when the query returned more than MAX_ROWS rows


def to_plain(value):
    """Convert a Snowflake value to a plain one that the checkpointer can store."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return str(value)


class QueryRunner:
    """Runs generated SELECT queries on the census data (the connection's role is read-only)."""

    def __init__(self, connection: SnowflakeConnection) -> None:
        self._connection = connection

    def run(self, sql: str) -> QueryResult:
        """Execute the query and return at most MAX_ROWS rows. Raises `QueryError`."""
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(sql, timeout=TIMEOUT_SECONDS)
                rows = cursor.fetchmany(MAX_ROWS + 1)
                columns = [column.name for column in cursor.description or []]
        except snowflake.connector.Error as exc:
            raise QueryError(str(exc)) from exc

        return QueryResult(
            columns=columns,
            rows=[[to_plain(value) for value in row] for row in rows[:MAX_ROWS]],
            truncated=len(rows) > MAX_ROWS,
        )
