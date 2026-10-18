from dataclasses import dataclass

import snowflake.connector
from snowflake.connector import SnowflakeConnection

from us_population_agent.config import Settings


class ColumnCatalogError(Exception):
    """Raised when the columns of a table cannot be read."""


MEANING_MARKER = " | Meaning: "


@dataclass(frozen=True)
class Column:
    """One column of a census data table."""

    name: str  # e.g. B01001e3
    meaning: str  # the part of the description specific to this column


@dataclass(frozen=True)
class TableColumns:
    """The columns of one census table, with the description part they all share."""

    data_table_name: str  # e.g. 2020_CBG_B01
    prefix: str  # table, topic, title and universe
    columns: list[Column]


def split_description(description: str) -> tuple[str, str]:
    """Splits `Column: X | <prefix> | Meaning: <meaning>` into its prefix and meaning.

    A description in another format is returned whole as the meaning, so nothing is lost.

    Args:
        description: Raw column description from the metadata table.

    Returns:
        A `(prefix, meaning)` tuple. The prefix is empty if the description has no meaning marker.
    """
    head, marker, meaning = description.partition(MEANING_MARKER)
    if not marker:
        return "", description
    return head.partition(" | ")[2], meaning


class ColumnCatalog:
    """Hard lookup of the columns that belong to a census table, from the metadata table."""

    def __init__(self, connection: SnowflakeConnection, settings: Settings) -> None:
        """Initializes the catalog.

        Args:
            connection: Open Snowflake connection used for all lookups.
            settings: Application settings providing the name of the columns table.
        """
        self._connection = connection
        # The table name comes from our own settings, not from user input; only the table id is bound.
        self._query = (
            f"SELECT COLUMN_NAME, DATA_TABLE_NAME, COLUMN_DESCRIPTION FROM {settings.snowflake_columns_table} "
            "WHERE TABLE_ID = %s ORDER BY LENGTH(COLUMN_NAME), COLUMN_NAME"
        )

    def columns_for(self, table_id: str) -> TableColumns | None:
        """Returns the columns of one table in numeric order.

        Args:
            table_id: Census table id, e.g. `B01001`.

        Returns:
            The table's columns with their shared description prefix, or None if the table has no columns.

        Raises:
            ColumnCatalogError: If the query to Snowflake fails.
        """
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(self._query, (table_id,))
                rows = cursor.fetchall()
        except snowflake.connector.Error as exc:
            raise ColumnCatalogError(f"Reading columns of {table_id} failed: {exc}") from exc

        if not rows:
            return None

        # The prefix is the same in every row of a table, so the first row is enough.
        prefix, _ = split_description(str(rows[0][2]))
        columns = [
            Column(name=str(name), meaning=split_description(str(description))[1]) for name, _, description in rows
        ]
        return TableColumns(data_table_name=str(rows[0][1]), prefix=prefix, columns=columns)
