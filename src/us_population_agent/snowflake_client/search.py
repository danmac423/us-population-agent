import logging
from dataclasses import dataclass
from typing import Any

import snowflake.connector
from snowflake.connector import SnowflakeConnection
from snowflake.core import Root
from snowflake.core.exceptions import SnowflakePythonError

from us_population_agent.config import Settings

logger = logging.getLogger(__name__)

SEARCH_COLUMNS = ["TABLE_ID", "DATA_TABLE_NAME", "TABLE_TITLE", "SEARCH_TEXT"]


class CatalogSearchError(Exception):
    """Raised when the catalog search fails or answers in an unexpected format."""


@dataclass(frozen=True)
class TableHit:
    """One census table returned by the catalog search."""

    table_id: str
    data_table_name: str
    title: str
    search_text: str
    reranker_score: float | None


def parse_hit(row: dict[str, Any]) -> TableHit | None:
    """Turns one result row into a `TableHit`.

    Args:
        row: A single result row from Cortex Search, including the optional `@scores` entry.

    Returns:
        The parsed hit, or None if the table id or data table name is missing.
    """
    table_id = row.get("TABLE_ID")
    data_table_name = row.get("DATA_TABLE_NAME")
    if not table_id or not data_table_name:
        return None

    scores = row.get("@scores")
    reranker_score = scores.get("reranker_score") if isinstance(scores, dict) else None
    return TableHit(
        table_id=str(table_id),
        data_table_name=str(data_table_name),
        title=str(row.get("TABLE_TITLE") or ""),
        search_text=str(row.get("SEARCH_TEXT") or ""),
        reranker_score=reranker_score if isinstance(reranker_score, int | float) else None,
    )


class CatalogSearch:
    """Cortex Search over the catalog of ACS tables."""

    def __init__(self, connection: SnowflakeConnection, settings: Settings) -> None:
        """Initializes the search client.

        Args:
            connection: Open Snowflake connection used to reach the Cortex Search service.
            settings: Application settings providing the service location and the result limit.
        """
        self._limit = settings.search_limit
        self._service = (
            Root(connection)
            .databases[settings.snowflake_database]
            .schemas[settings.snowflake_schema]
            .cortex_search_services[settings.snowflake_search_service]
        )

    def search(self, query: str) -> list[TableHit]:
        """Finds the census tables that best match a query.

        The SDK already validates that `results` is a list of dicts, so only the content of each row
        needs checking here. Rows with missing fields are skipped and logged.

        Args:
            query: Free-text search query.

        Returns:
            Matching tables, best first, up to the configured search limit.

        Raises:
            CatalogSearchError: If the search request fails.
        """
        try:
            response = self._service.search(query=query, columns=SEARCH_COLUMNS, limit=self._limit)
        except (SnowflakePythonError, snowflake.connector.Error) as exc:
            raise CatalogSearchError(f"Cortex Search request failed: {exc}") from exc

        hits = []
        for row in response.results:
            hit = parse_hit(row)
            if hit is None:
                logger.warning("Skipping malformed search result: %r", row)
                continue
            hits.append(hit)
        return hits
