import logging
from itertools import zip_longest
from typing import Protocol

from us_population_agent.agent.state import AgentState
from us_population_agent.snowflake_client.search import TableHit

logger = logging.getLogger(__name__)


class TableSearcher(Protocol):
    """What the search node needs from the catalog; `CatalogSearch` implements it."""

    def search(self, query: str) -> list[TableHit]: ...


MAX_CANDIDATES = 8


def merge_rankings(rankings: list[list[TableHit]], max_candidates: int = MAX_CANDIDATES) -> list[dict]:
    """Interleave the per-query rankings, drop duplicate tables and keep at most `max_candidates`.

    Rank 1 of every query comes first, then rank 2 and so on, so every topic of the question is represented
    before the cap cuts the list. Reranker scores are only comparable within one query, so they are not used.
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


def build_search(catalog: TableSearcher):
    """Create the search node: runs every rewritten query one after another and merges the results.

    A failing query is skipped. `search_failed` is set only when every query failed, so later nodes
    can tell "nothing found" from "the search is down".
    """

    def run_search(state: AgentState) -> dict:
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

        candidates = merge_rankings(rankings)
        logger.info("Search: %d queries, %d failed, %d candidates", len(queries), failures, len(candidates))
        return {"candidates": candidates, "search_failed": failures == len(queries)}

    return run_search
