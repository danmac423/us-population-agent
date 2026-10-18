import logging
from typing import Protocol

from us_population_agent.agent.state import AgentState
from us_population_agent.snowflake_client.query import QueryError, QueryResult, QueryUnavailableError

logger = logging.getLogger(__name__)


class SqlExecutor(Protocol):
    """What the node needs from the query runner; `QueryRunner` implements it."""

    def run(self, sql: str) -> QueryResult: ...


def build_execute_sql(runner: SqlExecutor):
    """Create the node that runs the generated SQL and stores the rows in the state."""

    def execute_sql(state: AgentState) -> dict:
        sql = state.get("sql")
        if not sql:
            return {"sql_result": None, "sql_error": None}

        try:
            result = runner.run(sql)
        except QueryUnavailableError as exc:
            logger.error("Snowflake is unavailable: %s", exc)
            return {"sql_result": None, "sql_error": str(exc), "technical_failure": True}
        except QueryError as exc:
            logger.warning("Query failed: %s", exc)
            return {"sql_result": None, "sql_error": str(exc)}
        except Exception as exc:
            # Not a Snowflake rejection of the SQL, so rewriting the query would not help.
            logger.exception("Unexpected error while running the query")
            return {"sql_result": None, "sql_error": str(exc) or type(exc).__name__, "technical_failure": True}

        logger.info("Query returned %d rows", len(result.rows))
        return {
            "sql_result": {"columns": result.columns, "rows": result.rows},
            "sql_error": None,
        }

    return execute_sql
