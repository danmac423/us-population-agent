import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from us_population_agent.agent.context import format_context
from us_population_agent.agent.schemas import TableSelection
from us_population_agent.agent.state import AgentState

logger = logging.getLogger(__name__)

SELECT_TABLES_SYSTEM_PROMPT = """# ROLE
You choose the census table that best answers the user's question. You get the question, notes from
the query rewriter and a list of candidate tables from the catalog of US Census (ACS) tables.
Do not answer the question and do not write SQL. Only choose a table.

# HOW A CANDIDATE IS DESCRIBED
Each candidate has a TABLE_ID, a title and a description with: topics, universe (who or what is counted)
and breakdowns (the categories the table is split into, level by level).

# HOW TO CHOOSE
1. The breakdowns must contain the categories the question needs. A threshold such as "under 18" is
   answered by summing the age groups below it, so a table split by age groups is a good fit.
2. The universe must match the population in the question. Do not pick a narrower table when the question
   is about everyone: avoid universes such as "citizens 18 years and over", "civilian employed population" or
   "veterans" unless the question is about exactly that group.
3. Prefer the simplest table that answers the question over one with extra splits (for example "Sex By Age"
   rather than "Sex By Age By Veteran Status") unless the question asks for those splits.
4. Use the rewriter notes (for example "age of the householder, not of all residents") to pick between
   similar tables.
5. If several tables fit, pick the most general one.
6. If no candidate can answer the question, return null and explain what is missing.
7. Only one table is used. If the question has several parts and the chosen table covers only some of them
   (for example it asks for income and education and you chose an income table), choose the table for the main
   part and list the other parts in "unanswered".

# OUTPUT
Fill in "reason" first (one or two sentences comparing the best candidates), then "table_id" with the TABLE_ID
of the chosen candidate. Use a TABLE_ID exactly as listed. Then fill in "unanswered" with the parts of the
question the chosen table cannot answer, each in a few plain words (empty list if it covers everything or
table_id is null)."""


def format_candidates(candidates: list[dict]) -> str:
    return "\n\n".join(
        f"TABLE_ID: {candidate['table_id']}\nTitle: {candidate['title']}\nDescription: {candidate['search_text']}"
        for candidate in candidates
    )


def build_select_tables(llm: BaseChatModel):
    """Create the node that picks the best table among the search candidates."""
    selector = llm.with_structured_output(TableSelection)

    def select_tables(state: AgentState) -> dict:
        candidates = state.get("candidates") or []
        *history, latest = state["messages"]

        if not candidates:
            return {"selected_table_id": None, "selection_reason": "No candidate tables.", "selection_unanswered": []}

        prompt = (
            f"Conversation so far:\n{format_context(history)}\n\n"
            f"Question: {latest.text}\n\n"
            f"Rewriter notes: {state.get('rewrite_notes') or 'none'}\n\n"
            f"Candidate tables:\n\n{format_candidates(candidates)}"
        )
        try:
            result = selector.invoke([SystemMessage(SELECT_TABLES_SYSTEM_PROMPT), HumanMessage(prompt)])
        except Exception:
            logger.exception("Table selection failed")
            return {
                "selected_table_id": None,
                "selection_reason": "The table selection failed.",
                "selection_unanswered": [],
            }

        if not isinstance(result, TableSelection):
            logger.error("Unexpected selector output: %r", result)
            return {
                "selected_table_id": None,
                "selection_reason": "The table selection failed.",
                "selection_unanswered": [],
            }

        # The model may invent an id, so accept only one that is really among the candidates.
        table_id = result.table_id if result.table_id in {c["table_id"] for c in candidates} else None
        if table_id is None and result.table_id is not None:
            logger.warning("Selector returned %r, which is not a candidate", result.table_id)
        logger.info("Selected table: %s (%s)", table_id, result.reason)
        return {
            "selected_table_id": table_id,
            "selection_reason": result.reason,
            "selection_unanswered": result.unanswered if table_id else [],
        }

    return select_tables
