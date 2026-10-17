import logging
from typing import Protocol

from us_population_agent.agent.state import AgentState
from us_population_agent.snowflake_client.columns import ColumnCatalogError, TableColumn

logger = logging.getLogger(__name__)


class ColumnSource(Protocol):
    """What the node needs from the column metadata; `ColumnCatalog` implements it."""

    def columns_for(self, table_id: str) -> list[TableColumn]: ...


def build_load_columns(columns: ColumnSource):
    """Create the node that loads the columns of the selected table (a plain lookup, no LLM)."""

    def load_columns(state: AgentState) -> dict:
        table_id = state.get("selected_table_id")
        if not table_id:
            return {"data_table_name": None, "table_columns": []}

        try:
            found = columns.columns_for(table_id)
        except ColumnCatalogError:
            logger.exception("Loading columns of %s failed", table_id)
            return {"data_table_name": None, "table_columns": []}

        data_table_name = found[0].data_table_name if found else None
        logger.info("Columns of %s: %d (data table %s)", table_id, len(found), data_table_name)
        return {
            "data_table_name": data_table_name,
            "table_columns": [{"name": column.column_name, "meaning": column.meaning} for column in found],
        }

    return load_columns
