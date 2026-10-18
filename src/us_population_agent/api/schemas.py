from pydantic import BaseModel, Field

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,100}$"  # stored by the checkpointer; keep it short and plain


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN, description="omit to start a new chat")


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
