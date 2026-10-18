from typing import cast

from us_population_agent.agent.graph import MAX_REPAIRS, route_after_execute_sql, route_after_validate_sql
from us_population_agent.agent.state import AgentState


def state(**fields) -> AgentState:
    return cast(AgentState, fields)


def test_valid_sql_is_executed():
    assert route_after_validate_sql(state(sql_errors=[])) == "execute_sql"


def test_invalid_sql_is_repaired_until_attempts_run_out():
    assert route_after_validate_sql(state(sql_errors=["x"], sql_attempts=0)) == "repair_sql"
    assert route_after_validate_sql(state(sql_errors=["x"], sql_attempts=MAX_REPAIRS)) == "no_sql"


def test_execution_error_is_repaired_until_attempts_run_out():
    assert route_after_execute_sql(state(sql_error=None)) == "answer"
    assert route_after_execute_sql(state(sql_error="boom", sql_attempts=1)) == "repair_sql"
    assert route_after_execute_sql(state(sql_error="boom", sql_attempts=MAX_REPAIRS)) == "no_sql"
