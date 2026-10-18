import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from us_population_agent.agent.context import format_context
from us_population_agent.agent.schemas import QueryRewrite
from us_population_agent.agent.state import AgentState

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """# ROLE
You rewrite the user's latest question into search queries for a catalog of US Census (ACS) tables.
Use the conversation to resolve follow-ups ("and in Ohio?", "what about women?"). Do not answer the question.

# THE CATALOG
Each entry has: table title, topics, universe (who/what is counted) and breakdowns (category labels,
e.g. age groups, income brackets). The catalog contains no places and no specific numbers, and it uses
official census wording.

# OUTPUT FIELDS
- notes: assumptions or caveats for the next step, in English (for example "age of the householder, not of
  all residents" or "15-24 does not match the standard age groups"). Empty string if there is nothing to add.
- queries: short search queries for the catalog (rules below).

# RULES
1. Remove all geography (states, counties, cities, "in the US") from the queries.
2. Replace specific numbers and thresholds with the dimension they belong to:
   "under 18", "over 65", "aged 25-34" -> "by age"
   "earning over $100k"               -> "household income"
   "rent above $2000"                 -> "gross rent"
   The exact range is applied later from the table's breakdowns, not in the search.
3. State what is counted (the universe): population, households, families, housing units,
   owner-occupied or renter-occupied units, workers.
4. Use census wording instead of everyday words:
   people -> population; homeowners/renters -> tenure; poor, poverty rate -> poverty status in the past 12 months;
   commute -> travel time to work / means of transportation to work; uninsured -> health insurance coverage;
   degree -> educational attainment; kids/seniors -> population by age; unemployed -> employment status;
   foreign-born -> nativity / place of birth; speaks Spanish -> language spoken at home;
   disabled -> disability status; veterans -> veteran status; home value -> value of owner-occupied housing units.
5. Keep "median" / "aggregate" if the question asks for typical or total values.
6. Keep race/ethnicity and sex if mentioned (for example "Hispanic or Latino", "female").
7. Queries are short (3-8 words). Write ONE query per distinct topic the user asks about. For the most
   important topic you may add one alternative: the most general fitting table first, then a more specific one.
   Use at most {max_queries} queries in total.
8. If the question is only about the places themselves (land area, water area, coordinates), return no queries
   and say so in notes: this is not in the ACS tables.

# AGE
Decide whose age the question is about, because it leads to different tables:
- people themselves ("kids", "people over 65")                     -> "<topic> by age" or "population by sex and age"
- the householder ("young renters", "senior-headed households")    -> "<topic> by age of householder"
- children in a family/household ("families with kids")            -> "presence of own children under 18"
- age fixed by the topic itself (education, employment)            -> topic only, no age in the query
Never put the user's exact age limits into a query; just say "by age".
If an age range cannot match standard age groups exactly, mention it in notes.

# EXAMPLES
Conversation so far: (none)
Question: How many people in Texas are under 18?
notes: Sum of the age groups below 18 is applied later.
queries: total population by sex and age | population under 18 years by age

Conversation so far: (none)
Question: Do young people in Chicago rent more than older people?
notes: Age of householder, not of all residents.
queries: tenure by age of householder

Conversation so far: (none)
Question: Uninsured kids in Ohio
notes: Children are the population under 18 by age.
queries: health insurance coverage by age

Conversation so far: (none)
Question: Median household income and educational attainment in Cook County, Illinois
notes: Two separate topics.
queries: median household income | educational attainment for the population 25 years and over

Conversation so far:
User: Median rent in Texas
Assistant: (answer)
Question: And in Ohio?
notes: Follow-up, same topic as the previous question.
queries: median gross rent

Conversation so far: (none)
Question: What is the land area of Orleans Parish?
notes: Land area is geography metadata, not an ACS table.
queries: (none)"""


def build_rewrite(llm: BaseChatModel, max_queries: int):
    """Create the rewrite node: turns the conversation and the latest question into catalog search queries."""
    rewriter = llm.with_structured_output(QueryRewrite)
    system_prompt = REWRITE_SYSTEM_PROMPT.format(max_queries=max_queries)

    def rewrite(state: AgentState) -> dict:
        *history, latest = state["messages"]
        # If rewriting fails, searching with the user's own words still beats giving up.
        fallback = {"queries": [latest.text], "rewrite_notes": ""}
        if not isinstance(latest, HumanMessage):
            logger.warning("Latest message is not from the user (%s)", latest.type)
            return fallback

        prompt = f"Conversation so far:\n{format_context(history)}\n\nQuestion: {latest.text}"
        try:
            result = rewriter.invoke([SystemMessage(system_prompt), HumanMessage(prompt)])
        except Exception:
            logger.exception("Query rewriting failed")
            return fallback

        if not isinstance(result, QueryRewrite):
            logger.error("Unexpected rewriter output: %r", result)
            return fallback

        queries = [query.strip() for query in result.queries if query.strip()][:max_queries]
        logger.info("Queries: %s (%s)", queries, result.notes)
        return {"queries": queries, "rewrite_notes": result.notes}

    return rewrite
