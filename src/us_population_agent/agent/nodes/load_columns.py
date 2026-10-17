import logging
from typing import Protocol

from us_population_agent.agent.state import AgentState
from us_population_agent.snowflake_client.columns import ColumnCatalogError, TableColumns

logger = logging.getLogger(__name__)


class ColumnSource(Protocol):
    """What the node needs from the column metadata; `ColumnCatalog` implements it."""

    def columns_for(self, table_id: str) -> TableColumns | None: ...


def build_load_columns(columns: ColumnSource):
    """Create the node that loads the columns of the selected table (a plain lookup, no LLM)."""

    def load_columns(state: AgentState) -> dict:
        empty = {"data_table_name": None, "table_prefix": None, "table_columns": []}
        table_id = state.get("selected_table_id")
        if not table_id:
            return empty

        try:
            found = columns.columns_for(table_id)
        except ColumnCatalogError:
            logger.exception("Loading columns of %s failed", table_id)
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
