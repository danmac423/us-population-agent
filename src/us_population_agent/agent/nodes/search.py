import logging
from itertools import zip_longest
from typing import Protocol

from us_population_agent.agent.state import AgentState
from us_population_agent.snowflake_client.search import TableHit

logger = logging.getLogger(__name__)


class TableSearcher(Protocol):
    """What the search node needs from the catalog; `CatalogSearch` implements it."""

    def search(self, query: str) -> list[TableHit]:
        """Finds the census tables that best match a query.

        Args:
            query: Free-text search query.

        Returns:
            Matching tables, best first.
        """


def merge_rankings(rankings: list[list[TableHit]], max_candidates: int) -> list[dict]:
    """Interleaves the per-query rankings, drops duplicate tables and keeps at most `max_candidates`.

    Rank 1 of every query comes first, then rank 2 and so on, so every topic of the question is represented
    before the cap cuts the list. Reranker scores are only comparable within one query, so they are not used.

    Args:
        rankings: One ranked list of hits per search query.
        max_candidates: Maximum number of candidates to return.

    Returns:
        Candidate tables as dicts with `table_id`, `data_table_name`, `title` and `search_text`.
    """
    candidates: dict[str, dict] = {}
    for hits_at_rank in zip_longest(*rankings):
        for hit in hits_at_rank:
            if hit is None or hit.table_id in candidates:
                continue
            candidates[hit.table_id] = {
                "table_id": hit.table_id,
                "data_table_name": hit.data_table_name,
                "title": hit.title,
                "search_text": hit.search_text,
            }
            if len(candidates) == max_candidates:
                return list(candidates.values())
    return list(candidates.values())


def build_search(catalog: TableSearcher, max_candidates: int):
    """Creates the search node.

    The node runs every rewritten query one after another and merges the results. A failing query is
    skipped. `search_failed` is set only when every query failed, so later nodes can tell "nothing found"
    from "the search is down".

    Args:
        catalog: Search over the catalog of census tables.
        max_candidates: Maximum number of candidate tables passed on.

    Returns:
        The `run_search` node function.
    """

    def run_search(state: AgentState) -> dict:
        """Searches the catalog with each rewritten query and merges the results.

        Args:
            state: Current graph state. Reads `queries`.

        Returns:
            A state update with the merged `candidates` and `search_failed`.
        """
        queries = state.get("queries") or []
        if not queries:
            return {"candidates": [], "search_failed": False}

        rankings: list[list[TableHit]] = []
        failures = 0
        for query in queries:
            try:
                rankings.append(catalog.search(query))
            except Exception:
                failures += 1
                logger.exception("Catalog search failed for %r", query)

        candidates = merge_rankings(rankings, max_candidates)
        logger.info("Search: %d queries, %d failed, %d candidates", len(queries), failures, len(candidates))
        return {"candidates": candidates, "search_failed": failures == len(queries)}

    return run_search
