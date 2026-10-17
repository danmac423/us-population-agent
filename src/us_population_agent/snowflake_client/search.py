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
    """Turn one result row into a `TableHit`, or return None when required fields are missing."""
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
        self._service = (
            Root(connection)
            .databases[settings.snowflake_database]
            .schemas[settings.snowflake_schema]
            .cortex_search_services[settings.snowflake_search_service]
        )

    def search(self, query: str, limit: int = 5) -> list[TableHit]:
        """Return the best matching tables, best first. Rows with missing fields are skipped.

        Raises `CatalogSearchError` when the request fails. The SDK already validates that `results`
        is a list of dicts, so only the content of each row needs checking here.
        """
        try:
            response = self._service.search(query=query, columns=SEARCH_COLUMNS, limit=limit)
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
