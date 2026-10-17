from dataclasses import dataclass

import snowflake.connector
from snowflake.connector import SnowflakeConnection

from us_population_agent.config import Settings


class ColumnCatalogError(Exception):
    """Raised when the columns of a table cannot be read."""


@dataclass(frozen=True)
class TableColumn:
    """One column of a census data table with its human-readable meaning."""

    column_name: str
    data_table_name: str
    meaning: str


def parse_meaning(description: str) -> str:
    """Keep only the `Meaning:` part of the description (the rest repeats table-level info for every column)."""
    return description.split("Meaning:", 1)[-1].strip()


class ColumnCatalog:
    """Hard lookup of the columns that belong to a census table, from the metadata table."""

    def __init__(self, connection: SnowflakeConnection, settings: Settings) -> None:
        self._connection = connection
        # The table name comes from our own settings, not from user input; only the table id is bound.
        self._query = (
            f"SELECT COLUMN_NAME, DATA_TABLE_NAME, COLUMN_DESCRIPTION FROM {settings.snowflake_columns_table} "
            "WHERE TABLE_ID = %s ORDER BY LENGTH(COLUMN_NAME), COLUMN_NAME"
        )

    def columns_for(self, table_id: str) -> list[TableColumn]:
        """Return the columns of one table (e.g. `B01001`) in numeric order. Raises `ColumnCatalogError`."""
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(self._query, (table_id,))
                rows = cursor.fetchall()
        except snowflake.connector.Error as exc:
            raise ColumnCatalogError(f"Reading columns of {table_id} failed: {exc}") from exc

        return [
            TableColumn(column_name=str(name), data_table_name=str(data_table), meaning=parse_meaning(str(description)))
            for name, data_table, description in rows
        ]
