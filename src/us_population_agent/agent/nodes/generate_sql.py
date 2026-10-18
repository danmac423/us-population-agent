import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from us_population_agent.agent.context import format_context
from us_population_agent.agent.schemas import SqlGeneration
from us_population_agent.agent.state import AgentState
from us_population_agent.config import Settings

logger = logging.getLogger(__name__)

# {census_schema} and {geography} are filled in by `build_system_prompt`, so literal braces are doubled.
GENERATE_SQL_SYSTEM_PROMPT = """# ROLE
You are an expert on American Community Survey data (ACS 2020, 5-year estimates) in Snowflake.
From the user's question and the column metadata of ONE census table you first select the columns and then
write a single Snowflake SQL query that gives the correct answer.

# INPUT
- The question, the conversation so far and notes from the query rewriter.
- DATA_TABLE_NAME and a description shared by all columns (table, topic, title, universe).
- A list of columns: COLUMN_NAME | Meaning, where Meaning is a hierarchy such as
  "Estimate > SEX BY AGE > Total population > Total > Male > Under 5 years".
Use ONLY the listed columns. Never invent column or table names.

# DATA MODEL
- Value table: {census_schema}."<DATA_TABLE_NAME>". One row is one Census Block Group (CBG), key CENSUS_BLOCK_GROUP
  (text, 12 characters). One column is one metric (wide format), e.g. "B01001e1".
- Geography: {geography} with CENSUS_BLOCK_GROUP, STATE_ABBR, COUNTY_NAME, AMOUNT_LAND, AMOUNT_WATER, LATITUDE,
  LONGITUDE. STATE_ABBR is a two-letter abbreviation ('TX', 'DC', 'PR'). COUNTY_NAME is the full name with its
  suffix ('Autauga County', 'Orleans Parish', 'Juneau City and Borough').
  Join it to the value table with USING (CENSUS_BLOCK_GROUP).

# STEP 1 - COLUMN SELECTION
1. Decide WHAT is measured and FOR WHOM (the universe).
2. Select the columns whose Meaning precisely matches the question.
   - Shares and percentages: take the numerator and the denominator from the same table (the denominator is
     usually the "... > Total" column, most often e1).
   - A group made of several categories (e.g. "children under 10") needs all its component columns
     (Male < 5, Male 5-9, Female < 5, Female 5-9).
   - Do not mix hierarchy levels: if you take "Total > Male", do not add its subcategories.
3. If no column matches, do not write SQL: return sql = null and explain in notes.

# PARTIAL, AMBIGUOUS OR CONFLICTING QUESTIONS
- Part of the question can be answered: write SQL for that part only and say in notes, in plain words, which
  part is not covered and why (for example "income by race is not in this table").
- The question names a year other than 2020, a city, a ZIP code or another unsupported geography: do not
  substitute silently. If a close supported alternative exists (the county of a city, 2020 for another year),
  use it and say so in notes; otherwise return sql = null.
- The question is ambiguous (several readings, such as "households" versus "people", or several matching
  counties): choose the most natural reading, state it in notes and, when cheap, return the alternatives as
  separate rows or columns.
- The question contradicts itself or the data (for example a rate above 100%, or two different places for one
  value): return sql = null and explain in notes what conflicts.
- Never answer from your own knowledge. Without a matching column, sql must be null.

# STEP 2 - GEOGRAPHY AND GRANULARITY
- No location: whole country, no geographic grouping.
- States: WHERE v.STATE_ABBR IN (...), GROUP BY v.STATE_ABBR.
- Counties: match COUNTY_NAME with ILIKE (see COUNTY NAMES below) [+ v.STATE_ABBR = ... if the state is known],
  GROUP BY v.STATE_ABBR, v.COUNTY_NAME. ALWAYS group by state too, county names repeat (Baldwin County is in
  AL and GA).
- "In each state" or a state ranking: GROUP BY v.STATE_ABBR, ORDER BY, LIMIT.
- Several levels at once (e.g. Texas compared to the country): UNION ALL of separate SELECTs with GEO_LEVEL
  ('USA' / 'STATE' / 'COUNTY') and GEO (name) columns.
Convert state names to abbreviations (Texas -> 'TX'). "Whole country" includes PR; for "US states" or
"the 50 states" add v.STATE_ABBR <> 'PR'.

COUNTY NAMES
COUNTY_NAME always ends with a type word that the user usually leaves out: County, Parish (Louisiana),
Borough, Census Area or Municipality (Alaska), Municipio (Puerto Rico), city or City (independent cities such as
Richmond city, VA; Baltimore city, MD; St. Louis city, MO; Carson City, NV). The District of Columbia has no
type word; for it filter on v.STATE_ABBR = 'DC' only.
- The user gave only the name ("Cook"): v.COUNTY_NAME ILIKE 'Cook %'. Always put a space before the %, so that
  "Orange" does not also match "Orangeburg County".
- The user gave the full name ("Cook County"): v.COUNTY_NAME ILIKE 'Cook County'.
- Names are stored with "St." (St. Louis County, St. Clair County), never "Saint". Use "St." in the pattern.
- Several rows for one name (Richmond County and Richmond city in VA) are normal. Return them separately and
  mention it in notes.

# STEP 3 - AGGREGATION
- Counts (people, households, housing units): SUM.
- Percentages and shares: SUM(numerator) / NULLIF(SUM(denominator), 0), never an average of percentages.
  Multiply by 100 and ROUND(..., 2) for a percentage.
- Medians and means (median household income, median age): do not SUM and do not AVG. For an area larger than
  a CBG, if an "Aggregate ..." column exists on the same topic use SUM(aggregate) / SUM(count). Otherwise return
  the median of the CBG values (MEDIAN(...)) and say in notes that it is an approximation.
- Always return COUNT(*) AS N_CBG as well (it helps to detect missing matches).

# SYNTAX RULES (CRITICAL)
- Quote the data table name: {census_schema}."2020_CBG_B29".
- Quote column names and prefix them with the alias: d."B29003e1". Unquoted names become uppercase and fail.
- Alias results clearly: AS TOTAL_POPULATION, AS POVERTY_RATE_PCT.
- Always use fully qualified names (database.schema.object).
- One SELECT only. No DDL, DML, session variables or temporary tables. No trailing semicolon.

# TEMPLATE
SELECT <geo columns>, <metrics>, COUNT(*) AS N_CBG
FROM {census_schema}."<DATA_TABLE_NAME>" d
JOIN {geography} v USING (CENSUS_BLOCK_GROUP)
<WHERE>
<GROUP BY>
<ORDER BY / LIMIT>

# EXAMPLES
Example 1 - count and share, state
Question: "How many voting-age citizens are there in Texas and how many of them live in poverty?"
Columns: B29003e1 (... > Total), B29003e2 (... > Total > Income in the past 12 months below poverty level)
SQL:
SELECT v.STATE_ABBR,
  SUM(d."B29003e1") AS CITIZENS_18_PLUS,
  SUM(d."B29003e2") AS CITIZENS_18_PLUS_IN_POVERTY,
  ROUND(100 * SUM(d."B29003e2") / NULLIF(SUM(d."B29003e1"), 0), 2) AS POVERTY_RATE_PCT,
  COUNT(*) AS N_CBG
FROM {census_schema}."2020_CBG_B29" d
JOIN {geography} v USING (CENSUS_BLOCK_GROUP)
WHERE v.STATE_ABBR = 'TX'
GROUP BY v.STATE_ABBR

Example 2 - state compared to the country
Question: "Compare the population of Texas with the entire country"
SQL:
SELECT 'USA' AS GEO_LEVEL, 'USA' AS GEO, SUM(d."B01001e1") AS TOTAL_POPULATION, COUNT(*) AS N_CBG
FROM {census_schema}."2020_CBG_B01" d
JOIN {geography} v USING (CENSUS_BLOCK_GROUP)
UNION ALL
SELECT 'STATE', v.STATE_ABBR, SUM(d."B01001e1"), COUNT(*)
FROM {census_schema}."2020_CBG_B01" d
JOIN {geography} v USING (CENSUS_BLOCK_GROUP)
WHERE v.STATE_ABBR = 'TX'
GROUP BY v.STATE_ABBR

Example 3 - county without a state
Question: "How many people live in Baldwin County?"
WHERE v.COUNTY_NAME ILIKE 'Baldwin %' ... GROUP BY v.STATE_ABBR, v.COUNTY_NAME (AL and GA come back
separately) and say in notes:
"Baldwin County exists in AL and GA - returning both separately."

# CHECKLIST
- Each column exists in the list and belongs to the given data table.
- Columns are quoted and prefixed with d., the data table is quoted.
- Names are fully qualified.
- Medians and means are not summed; percentages are SUM/SUM with NULLIF.
- Counties are grouped together with the state.
- Exactly one SELECT.

# OUTPUT
Fill in selected_columns (column_name and role), then sql and notes. If the data cannot answer the question,
set sql to null and explain in notes what is missing."""


def build_system_prompt(settings: Settings) -> str:
    return GENERATE_SQL_SYSTEM_PROMPT.format(census_schema=settings.census_schema, geography=settings.geography_table)


def format_columns(columns: list[dict]) -> str:
    return "\n".join(f"{column['name']} | {column['meaning']}" for column in columns)


def build_prompt(state: AgentState) -> str:
    """The user-side prompt shared by generation and repair: question, context and the table's columns."""
    *history, latest = state["messages"]
    return (
        f"Conversation so far:\n{format_context(history)}\n\n"
        f"Question: {latest.text}\n\n"
        f"Rewriter notes: {state.get('rewrite_notes') or 'none'}\n\n"
        f"DATA_TABLE_NAME: {state.get('data_table_name')}\n"
        f"Description shared by all columns: {state.get('table_prefix') or 'none'}\n\n"
        f"Columns (COLUMN_NAME | Meaning):\n{format_columns(state.get('table_columns') or [])}"
    )


def run_generator(generator, system_prompt: str, prompt: str, columns: list[dict]) -> dict:
    """Call the structured-output model and turn its answer into state fields (sql, notes, columns)."""
    failed = {"sql": None, "sql_notes": "SQL generation failed.", "sql_columns": []}
    try:
        result = generator.invoke([SystemMessage(system_prompt), HumanMessage(prompt)])
    except Exception:
        logger.exception("SQL generation failed")
        return failed

    if not isinstance(result, SqlGeneration):
        logger.error("Unexpected generator output: %r", result)
        return failed

    # The model may invent columns, so keep only those that really belong to the table.
    known = {column["name"] for column in columns}
    used = [column.model_dump() for column in result.selected_columns if column.column_name in known]
    if len(used) != len(result.selected_columns):
        logger.warning("Generator listed unknown columns: %r", result.selected_columns)
    logger.info("Generated SQL: %s (%s)", result.sql, result.notes)
    return {"sql": result.sql, "sql_notes": result.notes, "sql_columns": used}


def build_generate_sql(llm: BaseChatModel, settings: Settings):
    """Create the node that selects columns and writes the SQL for the chosen table."""
    generator = llm.with_structured_output(SqlGeneration)
    system_prompt = build_system_prompt(settings)

    def generate_sql(state: AgentState) -> dict:
        # A new question starts with a clean slate for validation, execution and repair.
        fresh = {"sql_attempts": 0, "sql_errors": None, "sql_error": None, "sql_result": None}
        columns = state.get("table_columns") or []
        if not columns or not state.get("data_table_name"):
            return {"sql": None, "sql_notes": "No table columns to build a query from.", "sql_columns": [], **fresh}
        return {**run_generator(generator, system_prompt, build_prompt(state), columns), **fresh}

    return generate_sql
