import logging

from langchain_core.language_models import BaseChatModel

from us_population_agent.agent.nodes.generate_sql import build_prompt, build_system_prompt, run_generator
from us_population_agent.agent.schemas import SqlGeneration
from us_population_agent.agent.state import AgentState
from us_population_agent.config import Settings

logger = logging.getLogger(__name__)


def build_repair_sql(llm: BaseChatModel, settings: Settings):
    """Creates the node that rewrites SQL which failed validation or execution.

    The model is given the problems found in the previous attempt.

    Args:
        llm: Strong chat model that rewrites the SQL.
        settings: Application settings used to fill in the system prompt.

    Returns:
        The `repair_sql` node function.
    """
    generator = llm.with_structured_output(SqlGeneration, method="json_schema")
    system_prompt = build_system_prompt(settings)

    def repair_sql(state: AgentState) -> dict:
        """Asks the model to fix the failing query and counts the attempt.

        Args:
            state: Current graph state. Reads the failing `sql`, its notes and the validation or execution errors.

        Returns:
            A state update with the new `sql`, its notes, an incremented `sql_attempts` and the
            validation and execution fields reset.
        """
        problems = state.get("sql_errors") or [state.get("sql_error") or "Unknown error."]
        prompt = (
            f"{build_prompt(state)}\n\n"
            "# PREVIOUS ATTEMPT FAILED\n"
            f"SQL:\n{state.get('sql')}\n\n"
            f"Notes you wrote: {state.get('sql_notes') or 'none'}\n\n"
            "Problems:\n" + "\n".join(f"- {problem}" for problem in problems) + "\n\n"
            "Fix these problems and return the corrected query. Keep what was already correct and the same reading "
            "of the question. Keep the notes accurate for the new query. If the problems cannot be fixed with the "
            "listed columns, set sql to null and explain in notes."
        )
        logger.info("Repairing SQL (attempt %d): %s", (state.get("sql_attempts") or 0) + 1, problems)
        return {
            **run_generator(generator, system_prompt, prompt, state.get("table_columns") or []),
            "sql_attempts": (state.get("sql_attempts") or 0) + 1,
            "sql_errors": None,
            "sql_error": None,
            "sql_result": None,
        }

    return repair_sql
