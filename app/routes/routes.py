from typing import List, Optional
from fastapi import APIRouter, Form, UploadFile
from app.models.models import QueryPromptRequest
from app.services.process import QueryProcessorService

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check():
    return {"data": {}, "message": "API is healthy", "status": 200}


@router.post("/query", tags=["Prompt"])
async def handle_query(
    query: str = Form(..., description="The prompt query string"),
    files: Optional[List[UploadFile]] = None
    ):
    try:
        # Placeholder for query handling logic
        request_data = QueryPromptRequest(prompt=query)
        query_processor = QueryProcessorService()
        response = await query_processor.process_query(request_data, files=files)
        print(f"Query processed successfully: {response}")
        return {"data":response, "status":200, "message":"Query processed successfully"}
    except Exception as e:
        print(f"Error in handle_query: {e}")
        return {"error": f"Failed to handle query: {e}"}