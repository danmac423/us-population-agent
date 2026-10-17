from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["answerable", "needs_clarification", "meta", "off_topic"]


class IntentDecision(BaseModel):
    """Output of the intent classifier; stored in the graph state as a whole."""

    # `reason` comes first so the model writes its justification before committing to a label.
    reason: str = Field(description="One or two sentences justifying the decision, see the OUTPUT section.")
    intent: Intent


class QueryRewrite(BaseModel):
    """Output of the query rewriter: search queries for the table catalog."""

    # `notes` comes first so assumptions are written before the queries that depend on them.
    notes: str = Field(description="Assumptions or caveats for the next step; empty string if none.")
    queries: list[str] = Field(
        description="Short catalog search queries (3-8 words each, at most 4), one per distinct topic."
    )


class TableSelection(BaseModel):
    """Output of the table selector: the candidate table that best answers the question."""

    # `reason` comes first so the model compares the candidates before committing to one.
    reason: str = Field(description="One or two sentences: why this table fits, or why none of the candidates does.")
    table_id: str | None = Field(
        description="TABLE_ID of the chosen candidate, or null when no candidate can answer the question."
    )
