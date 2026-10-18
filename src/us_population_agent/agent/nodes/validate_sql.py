import logging

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from us_population_agent.agent.state import AgentState
from us_population_agent.config import Settings

logger = logging.getLogger(__name__)

FORBIDDEN = {exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.Command, exp.Into}
GEOGRAPHY_COLUMNS = {
    "CENSUS_BLOCK_GROUP",
    "STATE_ABBR",
    "COUNTY_NAME",
    "AMOUNT_LAND",
    "AMOUNT_WATER",
    "LATITUDE",
    "LONGITUDE",
}


def validate_sql(sql: str, *, settings: Settings, data_table: str, columns: list[dict]) -> list[str]:
    """Finds the problems in the generated SQL without touching Snowflake.

    Args:
        sql: The generated SQL.
        settings: Application settings providing the schema and geography table names.
        data_table: Name of the data table chosen for the question.
        columns: Columns of the chosen table, each with a `name`.

    Returns:
        The problems found, or an empty list if the SQL is fine.
    """
    try:
        statements = sqlglot.parse(sql, dialect="snowflake")
    except SqlglotError as exc:
        return [f"The SQL does not parse: {exc}"]
    if len(statements) != 1 or statements[0] is None:
        return ["The query must be exactly one statement."]
    tree = statements[0]
    if not isinstance(tree, exp.Query) or any(tree.find(kind) for kind in FORBIDDEN):
        return ["Only a single SELECT query is allowed."]
    return check_tables(tree, settings, data_table) + check_columns(tree, {column["name"] for column in columns})


def check_tables(tree: exp.Query, settings: Settings, data_table: str) -> list[str]:
    """Checks that only the chosen data table and the geography table are used, fully qualified.

    Args:
        tree: Parsed query.
        settings: Application settings providing the schema and geography table names.
        data_table: Name of the data table chosen for the question.

    Returns:
        A problem for each disallowed table, and one if the data table is not used.
    """
    data = f"{settings.census_schema}.{data_table}".lower()
    geography = settings.geography_table.lower()
    used = {".".join(part.name for part in table.parts).lower() for table in tree.find_all(exp.Table)}
    errors = [
        f'Table "{name}" is not allowed. Use only {settings.census_schema}."{data_table}" and '
        f"{settings.geography_table}, fully qualified."
        for name in sorted(used - {data, geography})
    ]
    if data not in used:
        errors.append(f'The data table {settings.census_schema}."{data_table}" is not used.')
    return errors


def check_columns(tree: exp.Query, known: set[str]) -> list[str]:
    """Checks that census columns exist, are quoted and carry the table alias.

    Any other name must be a geography column or an alias defined in the query.

    Args:
        tree: Parsed query.
        known: Names of the columns of the chosen data table.

    Returns:
        A problem for each column that breaks these rules.
    """
    known_lower = {name.lower(): name for name in known}
    aliases = {alias.alias.upper() for alias in tree.find_all(exp.Alias)}
    errors = []
    for column in tree.find_all(exp.Column):
        name = column.name
        if name in known:
            if not column.this.quoted:
                errors.append(f'Column {name} must be quoted: d."{name}".')
            elif not column.table:
                errors.append(f'Column "{name}" must be prefixed with the table alias: d."{name}".')
        elif name.lower() in known_lower:
            errors.append(f'Column "{name}" does not exist; the listed name is "{known_lower[name.lower()]}".')
        elif name.upper() not in GEOGRAPHY_COLUMNS | aliases:
            errors.append(f'Column "{name}" is not in the column list or the geography table.')
    return errors


def build_validate_sql(settings: Settings):
    """Creates the node that checks the generated SQL without running it and stores the problems found.

    Args:
        settings: Application settings providing the schema and geography table names.

    Returns:
        The `validate` node function.
    """

    def validate(state: AgentState) -> dict:
        """Validates the generated SQL.

        Args:
            state: Current graph state. Reads `sql`, `data_table_name` and `table_columns`.

        Returns:
            A state update with `sql_errors`, empty if the SQL passed.
        """
        errors = validate_sql(
            state.get("sql") or "",
            settings=settings,
            data_table=state.get("data_table_name") or "",
            columns=state.get("table_columns") or [],
        )
        if errors:
            logger.warning("Generated SQL failed validation: %s", errors)
        return {"sql_errors": errors}

    return validate
