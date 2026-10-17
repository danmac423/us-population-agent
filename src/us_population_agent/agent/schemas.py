from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["answerable", "needs_clarification", "meta", "off_topic"]


class IntentDecision(BaseModel):
    """Output of the intent classifier; stored in the graph state as a whole."""

    # `reason` comes first so the model writes its justification before committing to a label.
    reason: str = Field(description="One or two sentences justifying the decision, see the OUTPUT section.")
    intent: Intent
