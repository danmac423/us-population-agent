import logging
from typing import Protocol

from us_population_agent.agent.state import AgentState
from us_population_agent.snowflake_client.columns import ColumnCatalogError, TableColumns

logger = logging.getLogger(__name__)


class ColumnSource(Protocol):
    """What the node needs from the column metadata; `ColumnCatalog` implements it."""

    def columns_for(self, table_id: str) -> TableColumns | None:
        """Returns the columns of one table.

        Args:
            table_id: Census table id, e.g. `B01001`.

        Returns:
            The table's columns, or None if the table has none.

        Raises:
            ColumnCatalogError: If the columns cannot be read.
        """


def build_load_columns(columns: ColumnSource):
    """Creates the node that loads the columns of the selected table.

    This is a plain lookup, with no LLM involved.

    Args:
        columns: Source of the column metadata.

    Returns:
        The `load_columns` node function.
    """

    def load_columns(state: AgentState) -> dict:
        """Loads the columns of the selected table into the state.

        Any failure, or a table without columns, yields an empty result instead of stopping the chat.

        Args:
            state: Current graph state. Reads `selected_table_id`.

        Returns:
            A state update with `data_table_name`, `table_prefix` and `table_columns`.
            `table_columns` is empty if nothing could be loaded.
        """
        empty = {"data_table_name": None, "table_prefix": None, "table_columns": []}
        table_id = state.get("selected_table_id")
        if not table_id:
            return empty

        try:
            found = columns.columns_for(table_id)
        except ColumnCatalogError:
            logger.exception("Loading columns of %s failed", table_id)
            return empty
        except Exception:
            # Anything the catalog did not wrap is still an infrastructure failure, not a reason to stop the chat.
            logger.exception("Unexpected error while loading columns of %s", table_id)
            return empty

        if found is None:
            logger.warning("Table %s has no columns in the metadata", table_id)
            return empty

        logger.info("Columns of %s: %d (data table %s)", table_id, len(found.columns), found.data_table_name)
        return {
            "data_table_name": found.data_table_name,
            "table_prefix": found.prefix,
            "table_columns": [{"name": column.name, "meaning": column.meaning} for column in found.columns],
        }

    return load_columns
