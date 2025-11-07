from pydantic import BaseModel

class QueryPromptRequest(BaseModel):
    prompt: str