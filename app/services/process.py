import sys
from typing import List, Optional
from fastapi import UploadFile
from app.models.models import QueryPromptRequest
from app.services.blobservice import upload_blob
from app.services.rag_system import RAGSystem

class QueryProcessorService:
    print("Initializing RAG System...")
    def __init__(self):
        try:
            self.rag_system = RAGSystem()
            print("System initialized successfully!\n")
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            sys.exit(1)
    
    async def process_query(self, query: QueryPromptRequest, files: Optional[List[UploadFile]] = None):
        try:
            # Implement your query processing logic here
            print(f"Processing query with prompt: ", query)
            prompt = query.prompt

            # for file in files or []:
            if files:
                file_uploaded_response = await upload_blob(files)
            
            # pass the query to rag system to process and get response
            rag_response = self.rag_system.answer_question(prompt)
            print(f"RAG System response: ", rag_response)

            return {"message": f"Processed query: {prompt}", "upload_info": file_uploaded_response if files else None, "rag_response": rag_response}
        except Exception as e:
            print(f"Error processing query: {e}")
            return {"error": "Failed to process query"}