from typing import List, Optional
from fastapi import APIRouter, File, Form, UploadFile
from app.models.models import QueryPromptRequest
from app.services.process import QueryProcessorService
from app.log_config import logger

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


@router.post("/analyse", tags=["Assessment analyser"])
async def handle_assessment(
    prompt: str = Form(..., description="The assessment prompt string"),
    thread_id: Optional[str] = Form(None, description="Thread ID for conversation continuity"),
    files: Optional[List[UploadFile]] = File(None, description="Optional list of uploaded files")):
    logger.info(f"Received assessment request: {prompt}, thread_id: {thread_id}")
    try:
        request_data = QueryPromptRequest(prompt=prompt, thread_id=thread_id)
        logger.info(f"Assessment request received - {request_data}")
        response = await QueryProcessorService().process_assessment(request_data, files=files)
        logger.info(f"Assessment processed successfully: {response}")
        return {"data": response, "status": 200, "message": "Assessment handled successfully"}
    except Exception as e:
        logger.error(f"Error in handle_assessment: {e}")
        return {"error": f"Failed to handle assessment: {e}"}
    
@router.get("/test-sentry", tags=["Testing"])
def test_sentry_integration():
    try:
        1 / 0  # This will raise a ZeroDivisionError
    except ZeroDivisionError as e:
        logger.error("Testing Sentry integration: Division by zero error", exc_info=e)
        raise e  # Re-raise the exception to be captured by Sentry