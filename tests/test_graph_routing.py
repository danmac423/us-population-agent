from typing import cast

from us_population_agent.agent.graph import (
    MAX_REPAIRS,
    no_sql_hint,
    no_table_hint,
    route_after_execute_sql,
    route_after_generate_sql,
    route_after_validate_sql,
)
from us_population_agent.agent.nodes.answer import unanswered_parts
from us_population_agent.agent.state import AgentState


def state(**fields) -> AgentState:
    return cast(AgentState, fields)


def test_valid_sql_is_executed():
    assert route_after_validate_sql(state(sql_errors=[])) == "execute_sql"


def test_invalid_sql_is_repaired_until_attempts_run_out():
    assert route_after_validate_sql(state(sql_errors=["x"], sql_attempts=0)) == "repair_sql"
    assert route_after_validate_sql(state(sql_errors=["x"], sql_attempts=MAX_REPAIRS)) == "failed_query"


def test_execution_error_is_repaired_until_attempts_run_out():
    assert route_after_execute_sql(state(sql_error=None)) == "answer"
    assert route_after_execute_sql(state(sql_error="boom", sql_attempts=1)) == "repair_sql"
    assert route_after_execute_sql(state(sql_error="boom", sql_attempts=MAX_REPAIRS)) == "failed_query"


def test_snowflake_outage_is_not_repaired():
    assert route_after_execute_sql(state(sql_error="down", technical_failure=True, sql_attempts=0)) == "unavailable"


def test_no_sql_means_no_data_unless_the_model_call_failed():
    assert route_after_generate_sql(state(sql="SELECT 1")) == "validate_sql"
    assert route_after_generate_sql(state(sql=None, technical_failure=False)) == "no_sql"
    assert route_after_generate_sql(state(sql=None, technical_failure=True)) == "unavailable"


def test_unanswered_parts_merge_both_stages_without_duplicates():
    parts = unanswered_parts(state(selection_unanswered=["education", " "], sql_unanswered=["education", "race"]))
    assert parts == ["education", "race"]
    assert unanswered_parts(state()) == []


def test_hints_carry_the_reasons_to_the_reply():
    assert no_table_hint(state(selection_reason="No fit.", rewrite_notes="Not an ACS topic.")) == (
        "No fit. Not an ACS topic."
    )
    assert no_table_hint(state()) is None
    assert no_sql_hint(state(sql_notes="Missing column.", sql_unanswered=["income by race"])) == (
        "Missing column. Not covered: income by race."
    )
