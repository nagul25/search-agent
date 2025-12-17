from datetime import datetime
import glob
import os
import shutil
import sys
from typing import List, Optional
from urllib.parse import urlparse
from fastapi import UploadFile
from app.models.models import QueryPromptRequest
from app.services.azure_assistant import run_assessment
from app.services.blobservice import download_blob_to_local, upload_blob, upload_png_to_blob
from app.services.pdf_conversion import convert_pdf_to_png
from app.services.rag_system import RAGSystem
from app.log_config import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        
    
    async def process_assessment(self, assessment: QueryPromptRequest, files: Optional[List[UploadFile]] = None):
        """
        Process an assessment request by:
        1. Uploading files to blob storage
        2. Converting PPT/PDF to PNG images
        3. Running the Azure Assistant with the images for assessment
        4. Returning the structured assessment result
        
        For follow-up questions, pass thread_id from previous response to maintain conversation history.
        """
        try:
            print(f"Processing assessment with prompt: ", assessment)
            prompt = assessment.prompt
            thread_id = assessment.thread_id  # For conversation continuity
            file_uploaded_response = None
            all_png_paths = []  # Collect all PNG paths for assistant analysis

            if files:
                # 1. Upload original files to blob storage (for backup/audit)
                file_uploaded_response = await upload_blob(files)

                # 2. Iterate through each file and download it for processing
                uploaded_files = file_uploaded_response.get("uploaded_files", [])
                
                for file_info in uploaded_files:
                    blob_url = file_info.get("blob_url")
                    logger.info(f"Downloading and processing file from blob url: {blob_url}")
                    
                    # Parse the url to get the file name
                    parsed_url = urlparse(blob_url)
                    file_name = parsed_url.path.split("/")[-1]
                    logger.info(f"Processing file: {file_name} from blob url: {blob_url}")

                    # 3. Download and write ppt/pdf slides into local path from azure
                    name, ext = os.path.splitext(file_name)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    local_path = os.path.join(PROJECT_ROOT, "tempfiles", name, f"{name}_{timestamp}{ext}")
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    await download_blob_to_local(blob_url, local_path)
                    logger.info(f"Downloaded file to local path: {local_path}")
                    print("input file extension: ", ext)
                    
                    # 4. Convert ppt/pdf to png
                    png_output_dir = convert_pdf_to_png(local_path, file_name=name)
                    logger.info(f"Converted PPT/PDF to PNGs at: {png_output_dir}")

                    # Collect PNG paths for assistant analysis
                    png_files_in_dir = sorted(glob.glob(os.path.join(png_output_dir, "*.png")))
                    all_png_paths.extend(png_files_in_dir)
                    logger.info(f"Found {len(png_files_in_dir)} PNG files for assessment")

                    # 5. Upload converted slides to blob as png (for backup/audit)
                    png_blob_files = await upload_png_to_blob(png_output_dir, file_name=name)
                    logger.info(f"Uploaded PNG files to blob storage: {png_blob_files}")

                    file_info["png_uploads"] = png_blob_files
                    
                    # 6. Remove the local original file after processing
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        logger.info(f"Removed local file: {local_path}")

            # 7. Run Azure Assistant for assessment
            logger.info(f"Running Azure Assistant assessment with {len(all_png_paths)} images")
            
            if all_png_paths:
                # Run assessment with images
                assessment_result = await run_assessment(
                    prompt=prompt,
                    png_paths=all_png_paths,
                    thread_id=thread_id
                )
            else:
                # Run assessment without images (text-only prompt)
                assessment_result = await run_assessment(
                    prompt=prompt,
                    png_paths=[],
                    thread_id=thread_id
                )
            
            logger.info(f"Assessment completed: success={assessment_result.get('success', False)}")

            # 8. Clean up PNG files and directories after assessment
            cleaned_dirs = set()
            for png_path in all_png_paths:
                try:
                    if os.path.exists(png_path):
                        os.remove(png_path)
                        # Track parent directory for cleanup
                        cleaned_dirs.add(os.path.dirname(png_path))
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up PNG file {png_path}: {cleanup_error}")
            
            # Clean up empty directories (slides dir and parent temp dir)
            for dir_path in cleaned_dirs:
                try:
                    if os.path.isdir(dir_path) and not os.listdir(dir_path):
                        shutil.rmtree(dir_path)
                        logger.info(f"Removed temporary PNG directory: {dir_path}")
                        # Also try to remove parent if empty
                        parent_dir = os.path.dirname(dir_path)
                        if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                            shutil.rmtree(parent_dir)
                            logger.info(f"Removed temporary parent directory: {parent_dir}")
                except Exception as dir_cleanup_error:
                    logger.warning(f"Failed to clean up directory {dir_path}: {dir_cleanup_error}")

            # Build response
            if assessment_result.get("success"):
                return {
                    "message": assessment_result.get("assessment"),
                    "assessment": assessment_result.get("assessment"),
                    "images_analyzed": assessment_result.get("images_analyzed", 0),
                    "thread_id": assessment_result.get("thread_id"),
                    "upload_info": file_uploaded_response
                }
            else:
                return {
                    "error": assessment_result.get("error", "Assessment failed"),
                    "upload_info": file_uploaded_response
                }

        except Exception as e:
            logger.error(f"Error processing assessment: {e}")
            return {"error": f"Failed to process assessment: {str(e)}"}