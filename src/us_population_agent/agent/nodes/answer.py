import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from us_population_agent.agent.context import format_context
from us_population_agent.agent.state import AgentState

logger = logging.getLogger(__name__)

FALLBACK_NO_DATA = (
    "I couldn't find a way to answer that from the census data. "
    "Could you rephrase the question or ask about a different measure or place?"
)
FALLBACK_ERROR = "Something went wrong while looking up the data. Please try again or rephrase the question."

ANSWER_SYSTEM_PROMPT = """# ROLE
You answer questions about the US population using the result of a SQL query on US Census (ACS 2020, 5-year
estimates) data. You get the question, the query result and notes about how it was computed.

# RULES
- Use ONLY numbers that appear in the query result. Never invent, estimate or recalculate beyond simple
  rounding or a comparison of the returned values.
- Start with the direct answer, then add short context if it helps. Be concise: a few sentences, plain text.
  For several rows (for example states or counties) use a short list.
- Format large numbers with thousands separators and percentages with a % sign.
- Mention the assumptions and approximations from the notes when they affect how the number should be read
  (for example "median is approximated from block-group medians", "Baldwin County exists in AL and GA").
- N_CBG is the number of census block groups behind a row. Do not mention it unless it is 0 or looks wrong.
  If you do, say "census block groups", never "N_CBG".
- If the notes say that an assumption was made (another year, a county instead of a city, one reading of an
  ambiguous question), say so clearly next to the number.
- If "Not answered" lists parts of the question, answer what the result covers, then state plainly which parts
  could not be answered here and offer to look them up in a separate question. Never guess those parts.
- If the result has no rows or the query failed, say plainly that no data was found. Do not guess a value.
  Use the notes to explain what is missing. Suggest how to retry: check the spelling of the place name, add the
  state (county names repeat across states), or try a broader area such as the state or the whole country.
- Never include technical identifiers or names in the answer: no SQL, table names, column names (such as
  B01003_001E or N_CBG), variable codes, census block group IDs, FIPS or GEOIDs, or internal step names. Describe
  everything in plain words (for example "total population", "median household income"). Name places by their
  common names, not codes. This applies even if the user asks for them.
- The data covers the US states, DC and Puerto Rico, at block-group level aggregated to county, state or country.
  Say so when the user asks about something outside the data."""


def unanswered_parts(state: AgentState) -> list[str]:
    """Parts of the question that table selection or SQL generation found uncovered, without duplicates."""
    parts = (state.get("selection_unanswered") or []) + (state.get("sql_unanswered") or [])
    return list(dict.fromkeys(part.strip() for part in parts if part.strip()))


def format_result(result: dict) -> str:
    header = " | ".join(result["columns"])
    rows = "\n".join(" | ".join(str(value) for value in row) for row in result["rows"])
    return f"{header}\n{rows}" if rows else f"{header}\n(no rows)"


def build_answer(llm: BaseChatModel):
    """Create the node that turns the query result into the final answer for the user."""

    def answer(state: AgentState) -> dict:
        *history, latest = state["messages"]
        result = state.get("sql_result")
        error = state.get("sql_error")

        if result is not None:
            outcome = f"Query result:\n{format_result(result)}"
        elif error:
            outcome = f"The query failed: {error}"
        else:
            outcome = "No query was run."

        unanswered = unanswered_parts(state)
        prompt = (
            f"Conversation so far:\n{format_context(history)}\n\n"
            f"Question: {latest.text}\n\n"
            f"Notes: {state.get('sql_notes') or 'none'}\n\n"
            f"Not answered: {'; '.join(unanswered) or 'none'}\n\n"
            f"{outcome}"
        )
        try:
            text = llm.invoke([SystemMessage(ANSWER_SYSTEM_PROMPT), HumanMessage(prompt)]).text.strip()
        except Exception:
            logger.exception("Generating the answer failed")
            text = ""

        if not text:
            # The user must never get an empty answer; show the raw result when we have one.
            text = f"Here is the result:\n{format_result(result)}" if result and result["rows"] else FALLBACK_NO_DATA
            if error and not result:
                text = FALLBACK_ERROR
            elif unanswered:
                text += f"\n\nI could not cover: {'; '.join(unanswered)}."
        return {"messages": [AIMessage(text)]}

    return answer
