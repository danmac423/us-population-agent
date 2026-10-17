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

# STEP 2 - GEOGRAPHY AND GRANULARITY
- No location: whole country, no geographic grouping.
- States: WHERE v.STATE_ABBR IN (...), GROUP BY v.STATE_ABBR.
- Counties: WHERE v.COUNTY_NAME IN (...) [+ state if given], GROUP BY v.STATE_ABBR, v.COUNTY_NAME.
  ALWAYS group by state too, county names repeat (Baldwin County is in AL and GA).
- "In each state" or a state ranking: GROUP BY v.STATE_ABBR, ORDER BY, LIMIT.
- Several levels at once (e.g. Texas compared to the country): UNION ALL of separate SELECTs with GEO_LEVEL
  ('USA' / 'STATE' / 'COUNTY') and GEO (name) columns.
Convert state names to abbreviations (Texas -> 'TX'). Add the "County" suffix when the user omitted it
(Louisiana uses "Parish", Alaska "Borough" or "Census Area"). "Whole country" includes PR; for "US states" or
"the 50 states" add v.STATE_ABBR <> 'PR'.

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
Group by v.STATE_ABBR, v.COUNTY_NAME (AL and GA come back separately) and say in notes:
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
    census_schema = f"{settings.snowflake_census_database}.{settings.snowflake_census_schema}"
    geography_schema = f"{settings.snowflake_database}.{settings.snowflake_schema}.{settings.snowflake_geography_table}"
    return GENERATE_SQL_SYSTEM_PROMPT.format(census_schema=census_schema, geography=geography_schema)


def format_columns(columns: list[dict]) -> str:
    return "\n".join(f"{column['name']} | {column['meaning']}" for column in columns)


def build_generate_sql(llm: BaseChatModel, settings: Settings):
    """Create the node that selects columns and writes the SQL for the chosen table."""
    generator = llm.with_structured_output(SqlGeneration)
    system_prompt = build_system_prompt(settings)

    def generate_sql(state: AgentState) -> dict:
        columns = state.get("table_columns") or []
        data_table_name = state.get("data_table_name")
        if not columns or not data_table_name:
            return {"sql": None, "sql_notes": "No table columns to build a query from.", "sql_columns": []}

        failed = {"sql": None, "sql_notes": "SQL generation failed.", "sql_columns": []}
        *history, latest = state["messages"]
        prompt = (
            f"Conversation so far:\n{format_context(history)}\n\n"
            f"Question: {latest.text}\n\n"
            f"Rewriter notes: {state.get('rewrite_notes') or 'none'}\n\n"
            f"DATA_TABLE_NAME: {data_table_name}\n"
            f"Description shared by all columns: {state.get('table_prefix') or 'none'}\n\n"
            f"Columns (COLUMN_NAME | Meaning):\n{format_columns(columns)}"
        )
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

    return generate_sql
