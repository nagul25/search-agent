from typing import Optional
from pydantic import BaseModel, Field

class QueryPromptRequest(BaseModel):
    prompt: str = Field(..., description="The prompt text submitted by the user")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")