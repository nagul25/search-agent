"""
Azure Assistants API Integration Module

This module provides functionality to interact with Azure OpenAI Assistants API
for architecture assessment using static standards (file_search) and user-uploaded
images (Vision capability).
"""

import os
import json
import time
import asyncio
import httpx
from typing import List, Optional, Dict, Any
from openai import AzureOpenAI
from app.log_config import logger

# Placeholder system prompt - Update this with your actual assessment bot instructions
ASSESSMENT_SYSTEM_PROMPT = """Role & Objective
You are an Enterprise Solution Architect at Experian, specializing in enterprise-scale technology solutions.
Your goal is to collect project information and then generate a Pre-Assessment Report based on Experian policies,
best practices, and uploaded references.

Phase 1: Requirements Gathering & Clarifications
1. Iterative Data Collection
o Actively prompt the user for any missing required data points, guided by:
? EA-Assessment-Categories
? Industry best practices
? User responses
o Do not repeatedly show a full summary of collected data. Instead, only highlight the missing required items that need to be provided.
2. Required and Optional Data Points
o Required Data:
? Contacts (Names & Roles)
? Organization (Functional Area)
? Impact Scope (Global/Regional)
? Project Name
? Project Description
? Nature of Project (New tools, Enhancement, Migration, M&A, POC)
? Data Classification (Public, Internal, Confidential, Restricted)
? Product Comparison (Options or alternatives considered)
? Technology Stacks (Tools, capabilities, SSO, API gateway, frameworks, EEC, etc.)
? Users (B2E, B2B, B2C, etc.)
? Project Timeline
o Optional Data:
? Design (Scalability, Extensibility, Maintainability, etc.)
? Strategies (Migration, Build vs. Buy, Cloud Adoption, Reuse, etc.)
? Observability (Logs, Monitoring)
? Interfaces (External, Internal, Integrations)
? Diagrams (Data Flow, Network, System)
? Proof-of-Concept (Default = No)
? Financial (Capital/OPEX Costs)
? Constraints & Dependencies
? Other Approvals (PSA, RSQ, GenAI Council)
? Other relevant technical data points
3. Minimal Screen Scrolling
o Keep prompts and confirmations concise.
o Present all information in a tight display (minimal line spacing).
o Only show the summary when the user explicitly requests it or once all required data is provided.
o Keep summary display data with minimal screen scrolling for the users in a tight display with minimal line spacing.
4. File Uploads & Extracting Data
o Users may upload diagrams or documents.
o Read and extract relevant technical details, even if not explicitly listed in the required or optional data points
(e.g., migration strategy, deployment approach, POC evaluation metrics, test plans, etc.).
o Automatically capture these details in the background without displaying them in full.
5. Highlight Gaps & Missing Data
o After each user response or file upload, highlight any required data still missing or any clarifications needed.
o Do not display a full summary of data collected each time—only the missing required data.
6. Completion Prompt
o Once all required data is collected, prompt the user:
1. "Would you like to see a summary of all collected data?"
2. "Proceed to Pre-Assessment Report?"
o If any data is still missing, note what it is in the summary.

Phase 2: Pre-Assessment Report
When the user chooses to proceed or requests the summary, generate a Pre-Assessment Report that includes:
1. Project Summary
o Project ID: Generate using ProjectName + [unique five-digit number based on date/time].
o Brief Description: Provide a short overview of the project.
o Nature of Project: (new capability, enhancement, migration, etc.)
2. Critical Gaps & Missing Data
o List any insufficient data that impacts an accurate assessment.

--------------------------------------------------
4. Assessment Section
Evaluate the project against these categories in detail and provide score and verbose evaluations, noting any insufficient data:
o Alignment with Experian EA/Cloud Principles
o Product Comparison (evaluation metrics, considered solutions)
? Score lower if no evaluation metrics is provided for selecting the tools or approach.
o Security & Compliance (data protection, access control, encryption, etc)
? Score lower if there are no integrations with Experian SSO/Okta or other security stacks and approved tools.
o Maintainability & Operability (support, interoperability, automation, etc)
? Score accordingly based on the tools or services or products information provided.
o Overall Design (scalability, resiliency, availability, performance, diagrams, etc)
? Score lower if diagrams are NOT provided or if details are not provided as to how the services are being deployed.
o Interfaces/Integrations (external/internal information)
o Portability (deployable to other environments)
o Observability (logging, monitoring, reporting, etc)
? Score higher if using Experian-approved tools or services for observability.
o Risks (security, compliance, technical)
? Score lower if lacking information.
o Overall Project Assessment (completeness of data and alignments)
? Provide information on how this aligns with the industry standards
5. Output format in a table format | Category | Score (0-5) | Detailed Evaluations and Justifications |
o Scoring Guide:
? 0 = No data provided
? 1-2 = Some data, but insufficient
? 3-4 = Mostly aligns with Experian and industry best practices.
? 5 = Fully meets standards/best practices
o Score higher when using approved Experian tools or frameworks.
o Always use the same scoring logic for fairness and consistency.
o Show the average project score at the end of the table.
6. Strengths & Weaknesses
o Summarize the project's technical strengths.
(Call out if the project or technology can be tagged as an Experian integration pattern that other Business partners can reuse.)
o List weaknesses (including any legacy tools).
7. Risks & Additional Considerations
o Highlight potential security, compliance, or other technical risks.
o If GenAI is involved, mention special security & compliance steps.
8. Next Steps & Recommendations
o Outline required approvals, extra documentation, or TEB/RSQ/PSA reviews.
o Indicate GenAI Council involvement if relevant.
o Suggest additional technical improvements (not repeating items already in weaknesses).
9. Final Assessment Reminder
o State: "This is a preliminary assessment; an EA will review it and confirm it; in the meantime,
| you can download the report for the analysis done."
o Provide a link to download the report.

Tone & Format Requirements
* Write concisely to reduce screen scrolling.
* Use headings, bullet points, tables to structure content.
* Maintain a professional but concise tone.
* Reference external best-practice frameworks as relevant.
"""


class AzureAssistantClient:
    """
    Client for interacting with Azure OpenAI Assistants API.
    
    Handles file uploads, assistant creation/retrieval, thread management,
    and response polling.
    """
    
    def __init__(self):
        """Initialize the Azure OpenAI client with credentials from environment."""
        self.endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
        self.api_key = os.getenv("AZURE_AI_FOUNDRY_KEY")
        self.api_version = os.getenv("AZURE_AI_FOUNDRY_API_VERSION", "2024-05-01-preview")
        self.model = os.getenv("AZURE_AI_FOUNDRY_DEPLOYMENT", "gpt-5")
        self.assistant_id = os.getenv("AZURE_ASSISTANT_ID")
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure AI Foundry credentials not configured. "
                "Please set AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_KEY environment variables."
            )
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            http_client=httpx.Client(verify=False)  # Disable SSL verification
        )
        logger.info("Azure Assistant Client initialized successfully")
    
    def upload_file(self, file_path: str, purpose: str = "assistants") -> str:
        """
        Upload a file to Azure OpenAI for use with Assistants API.
        
        Args:
            file_path: Path to the file to upload
            purpose: Purpose of the file (default: "assistants")
            
        Returns:
            The file ID of the uploaded file
        """
        try:
            with open(file_path, "rb") as file:
                response = self.client.files.create(
                    file=file,
                    purpose=purpose
                )
            logger.info(f"Uploaded file {file_path} with ID: {response.id}")
            return response.id
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {e}")
            raise
    
    def upload_files_batch(self, file_paths: List[str], purpose: str = "assistants") -> List[str]:
        """
        Upload multiple files to Azure OpenAI.
        
        Args:
            file_paths: List of file paths to upload
            purpose: Purpose of the files (default: "assistants")
            
        Returns:
            List of file IDs
        """
        file_ids = []
        for file_path in file_paths:
            file_id = self.upload_file(file_path, purpose)
            file_ids.append(file_id)
        return file_ids
    
    def create_assistant(
        self,
        name: str = "AssessmentAgent",
        instructions: str = ASSESSMENT_SYSTEM_PROMPT,
        file_ids: Optional[List[str]] = None,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Create a new assistant with file_search capability.
        
        For Azure OpenAI, files are attached at the thread/message level rather than
        the assistant level. The assistant is created with file_search tool enabled,
        and files are stored separately to be attached to each thread.
        
        Args:
            name: Name of the assistant
            instructions: System instructions for the assistant
            file_ids: List of file IDs (stored in config, attached at thread level)
            tools: List of tools to enable (default: file_search)
            
        Returns:
            The assistant ID
        """
        if tools is None:
            tools = [{"type": "file_search"}]
        
        try:
            tool_resources = None
            vector_store_id = None
            
            # Try to create vector store if file_ids provided (for OpenAI-compatible APIs)
            if file_ids:
                try:
                    # Check if vector_stores API is available
                    if hasattr(self.client.beta, 'vector_stores'):
                        vector_store = self.client.beta.vector_stores.create(
                            name=f"{name}_standards_store"
                        )
                        vector_store_id = vector_store.id
                        logger.info(f"Created vector store with ID: {vector_store_id}")
                        
                        # Add files to the vector store
                        self.client.beta.vector_stores.file_batches.create_and_poll(
                            vector_store_id=vector_store_id,
                            file_ids=file_ids
                        )
                        logger.info(f"Added {len(file_ids)} files to vector store")
                        
                        tool_resources = {
                            "file_search": {
                                "vector_store_ids": [vector_store_id]
                            }
                        }
                    else:
                        logger.info("Vector stores not available - files will be attached at thread level")
                except Exception as vs_error:
                    logger.warning(f"Vector store setup skipped: {vs_error}")
                    logger.info("Files will be attached at thread/message level instead")
            
            # Create the assistant (with or without tool_resources)
            create_params = {
                "name": name,
                "instructions": instructions,
                "model": self.model,
                "tools": tools
            }
            
            # Only add tool_resources if we successfully created a vector store
            if tool_resources:
                create_params["tool_resources"] = tool_resources
            
            assistant = self.client.beta.assistants.create(**create_params)
            logger.info(f"Created assistant with ID: {assistant.id}")
            
            if vector_store_id:
                logger.info(f"Assistant {assistant.id} linked to vector store {vector_store_id}")
            else:
                logger.info(f"Assistant {assistant.id} created - files will be attached per-thread")
            
            return assistant.id
            
        except Exception as e:
            logger.error(f"Failed to create assistant: {e}")
            raise
    
    def get_assistant(self, assistant_id: Optional[str] = None) -> Any:
        """
        Retrieve an existing assistant by ID.
        
        Args:
            assistant_id: The assistant ID (uses env var if not provided)
            
        Returns:
            The assistant object
        """
        aid = assistant_id or self.assistant_id
        if not aid:
            raise ValueError("No assistant ID provided or configured")
        
        try:
            assistant = self.client.beta.assistants.retrieve(aid)
            logger.info(f"Retrieved assistant: {assistant.id}")
            return assistant
        except Exception as e:
            logger.error(f"Failed to retrieve assistant {aid}: {e}")
            raise
    
    def create_thread_with_message(
        self,
        user_prompt: str,
        image_file_ids: Optional[List[str]] = None,
        standard_file_ids: Optional[List[str]] = None
    ) -> str:
        """
        Create a new thread with an initial user message, optional image attachments,
        and optional standard files for file_search.

        Azure OpenAI has a limit of 10 content items per message. If there are more
        than 9 images, the images are split across multiple messages (first message
        gets text + up to 9 images, subsequent messages get up to 10 images each).
        
        Args:
            user_prompt: The user's prompt/question
            image_file_ids: List of file IDs for image attachments (for vision)
            standard_file_ids: List of file IDs for standard docs (for file_search)
            
        Returns:
            The thread ID
        """
        # Azure OpenAI limits content array to 10 items per message
        MAX_CONTENT_ITEMS = 10
        # First message has text, so can only fit 9 images
        MAX_IMAGES_FIRST_MESSAGE = MAX_CONTENT_ITEMS - 1
        
        try:
            # Build attachments for file_search (standard documents)
            attachments = []
            if standard_file_ids:
                for file_id in standard_file_ids:
                    attachments.append({
                        "file_id": file_id,
                        "tools": [{"type": "file_search"}]
                    })
                logger.info(f"Attaching {len(standard_file_ids)} standard files for file_search")
            
            # Batch images if we have more than can fit in one message
            # All batches limited to 9 images since each includes a text element
            image_batches = []
            if image_file_ids:
                for i in range(0, len(image_file_ids), MAX_IMAGES_FIRST_MESSAGE):
                    image_batches.append(image_file_ids[i:i + MAX_IMAGES_FIRST_MESSAGE])
                
                logger.info(f"Split {len(image_file_ids)} images into {len(image_batches)} batches")
            
            # Build first message content (text + first batch of images)
            content = []
            content.append({
                "type": "text",
                "text": user_prompt
            })
            
            # Add first batch of images if available
            if image_batches:
                for file_id in image_batches[0]:
                    content.append({
                        "type": "image_file",
                        "image_file": {"file_id": file_id}
                    })
            
            # Create the first message
            message_params = {
                "role": "user",
                "content": content
            }
            
            # Add attachments if we have standard files
            if attachments:
                message_params["attachments"] = attachments
            
            # Create thread with first message
            thread = self.client.beta.threads.create(
                messages=[message_params]
            )
            logger.info(f"Created thread with ID: {thread.id}")

            # Add remaining image batches as separate messages
            if len(image_batches) > 1:
                for batch_idx, batch in enumerate(image_batches[1:], start=2):
                    batch_content = []
                    batch_content.append({
                        "type": "text",
                        "text": f"[Continued: Images batch {batch_idx} of {len(image_batches)}]"
                    })
                    for file_id in batch:
                        batch_content.append({
                            "type": "image_file",
                            "image_file": {"file_id": file_id}
                        })
                    
                    self.client.beta.threads.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=batch_content
                    )
                    logger.info(f"Added image batch {batch_idx}/{len(image_batches)} to thread")
            
            return thread.id
        except Exception as e:
            logger.error(f"Failed to create thread: {e}")
            raise
    
    def add_message_to_thread(
        self,
        thread_id: str,
        user_prompt: str,
        image_file_ids: Optional[List[str]] = None,
        standard_file_ids: Optional[List[str]] = None
    ) -> None:
        """
        Add a new user message to an existing thread for conversation continuity.

        Azure OpenAI has a limit of 10 content items per message. If there are more
        than 9 images, the images are split across multiple messages.
        
        Args:
            thread_id: The existing thread ID to add the message to
            user_prompt: The user's prompt/question
            image_file_ids: List of file IDs for image attachments (for vision)
            standard_file_ids: List of file IDs for standard docs (for file_search)
        """
        # Azure OpenAI limits content array to 10 items per message
        MAX_CONTENT_ITEMS = 10
        # First message has text, so can only fit 9 images
        MAX_IMAGES_FIRST_MESSAGE = MAX_CONTENT_ITEMS - 1
        
        try:
            # Build attachments for file_search (standard documents)
            attachments = []
            if standard_file_ids:
                for file_id in standard_file_ids:
                    attachments.append({
                        "file_id": file_id,
                        "tools": [{"type": "file_search"}]
                    })
                logger.info(f"Attaching {len(standard_file_ids)} standard files for file_search")
            
            # Batch images if we have more than can fit in one message
            # All batches limited to 9 images since each includes a text element
            image_batches = []
            if image_file_ids:
                for i in range(0, len(image_file_ids), MAX_IMAGES_FIRST_MESSAGE):
                    image_batches.append(image_file_ids[i:i + MAX_IMAGES_FIRST_MESSAGE])
                
                logger.info(f"Split {len(image_file_ids)} images into {len(image_batches)} batches")
            
            # Build first message content (text + first batch of images)
            content = []
            content.append({
                "type": "text",
                "text": user_prompt
            })
            
            # Add first batch of images if available
            if image_batches:
                for file_id in image_batches[0]:
                    content.append({
                        "type": "image_file",
                        "image_file": {"file_id": file_id}
                    })

            # Create the first message params
            message_params = {
                "role": "user",
                "content": content
            }
            
            # Add attachments if we have standard files
            if attachments:
                message_params["attachments"] = attachments
            
            # Add first message to thread
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                **message_params
            )
            logger.info(f"Added message to existing thread: {thread_id}")
            
            # Add remaining image batches as separate messages
            if len(image_batches) > 1:
                for batch_idx, batch in enumerate(image_batches[1:], start=2):
                    batch_content = []
                    batch_content.append({
                        "type": "text",
                        "text": f"[Continued: Images batch {batch_idx} of {len(image_batches)}]"
                    })
                    for file_id in batch:
                        batch_content.append({
                            "type": "image_file",
                            "image_file": {"file_id": file_id}
                        })
                    
                    self.client.beta.threads.messages.create(
                        thread_id=thread_id,
                        role="user",
                        content=batch_content
                    )
                    logger.info(f"Added image batch {batch_idx}/{len(image_batches)} to thread")

        except Exception as e:
            logger.error(f"Failed to add message to thread {thread_id}: {e}")
            raise
    
    async def run_assistant(
        self,
        thread_id: str,
        assistant_id: Optional[str] = None,
        poll_interval: float = 1.0,
        timeout: float = 300.0
    ) -> str:
        """
        Run the assistant on a thread and wait for completion.
        
        Args:
            thread_id: The thread ID to run on
            assistant_id: The assistant ID (uses env var if not provided)
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait for completion
            
        Returns:
            The assistant's response text
        """
        aid = assistant_id or self.assistant_id
        if not aid:
            raise ValueError("No assistant ID provided or configured")
        
        try:
            # Create the run
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=aid
            )
            logger.info(f"Created run with ID: {run.id}")
            
            # Poll for completion
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(f"Assistant run timed out after {timeout} seconds")
                
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                
                if run_status.status == "completed":
                    logger.info(f"Run {run.id} completed successfully")
                    break
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    error_msg = f"Run {run.id} ended with status: {run_status.status}"
                    if run_status.last_error:
                        error_msg += f" - {run_status.last_error.message}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                elif run_status.status == "requires_action":
                    logger.warning(f"Run {run.id} requires action - not implemented")
                    raise NotImplementedError("Tool calls requiring action not implemented")
                
                logger.info(f"Run status: {run_status.status}, elapsed: {elapsed:.1f}s, waiting...")
                await asyncio.sleep(poll_interval)
            
            # Get the response
            messages = self.client.beta.threads.messages.list(thread_id=thread_id)
            
            # Find the assistant's response (most recent assistant message)
            for message in messages.data:
                if message.role == "assistant":
                    # Extract text content
                    response_text = ""
                    for content_block in message.content:
                        if content_block.type == "text":
                            response_text += content_block.text.value
                    return response_text
            
            raise RuntimeError("No assistant response found in thread")
            
        except Exception as e:
            logger.error(f"Failed to run assistant: {e}")
            raise
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from Azure OpenAI.
        
        Args:
            file_id: The file ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            self.client.files.delete(file_id)
            logger.info(f"Deleted file: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    def delete_files_batch(self, file_ids: List[str]) -> Dict[str, bool]:
        """
        Delete multiple files from Azure OpenAI.
        
        Args:
            file_ids: List of file IDs to delete
            
        Returns:
            Dictionary mapping file IDs to deletion success status
        """
        results = {}
        for file_id in file_ids:
            results[file_id] = self.delete_file(file_id)
        return results


def load_standard_file_ids() -> List[str]:
    """
    Load standard file IDs from the standard_files.json config.
    
    Returns:
        List of file IDs for standard documents
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "standard_files.json"
    )
    
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                file_ids = config.get("files", [])
                if file_ids:
                    logger.info(f"Loaded {len(file_ids)} standard file IDs from config")
                return file_ids
    except Exception as e:
        logger.warning(f"Failed to load standard file IDs: {e}")
    
    return []


async def run_assessment(
    prompt: str,
    png_paths: List[str],
    assistant_id: Optional[str] = None,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a complete assessment flow using the Azure Assistant.
    
    This is the main entry point for processing assessments. It:
    1. Uploads user PNG images to Azure
    2. Loads standard file IDs from config
    3. Creates a new thread OR adds message to existing thread for conversation continuity
    4. Runs the assistant
    5. Cleans up uploaded user files
    6. Returns the assessment result
    
    Args:
        prompt: The user's assessment prompt
        png_paths: List of paths to PNG images to analyze
        assistant_id: Optional assistant ID (uses env var if not provided)
        thread_id: Optional thread ID for conversation continuity (reuses existing thread if provided)
        
    Returns:
        Dictionary containing the assessment result and metadata
    """
    client = AzureAssistantClient()
    uploaded_file_ids = []
    
    try:
        # Upload PNG files for vision analysis
        logger.info(f"Uploading {len(png_paths)} PNG files for assessment")
        for png_path in png_paths:
            if os.path.exists(png_path):
                file_id = client.upload_file(png_path, purpose="assistants")
                uploaded_file_ids.append(file_id)
            else:
                logger.warning(f"PNG file not found: {png_path}")
        
        if not uploaded_file_ids and png_paths:
            logger.warning("No PNG files were uploaded for assessment")
        
        # Load standard file IDs for file_search
        standard_file_ids = load_standard_file_ids()
        
        # Either reuse existing thread or create a new one
        if thread_id:
            # Add message to existing thread for conversation continuity
            logger.info(f"Adding message to existing thread: {thread_id}")
            client.add_message_to_thread(
                thread_id=thread_id,
                user_prompt=prompt,
                image_file_ids=uploaded_file_ids,
                standard_file_ids=standard_file_ids
            )
        else:
            # Create new thread with user message, image attachments, and standard files
            thread_id = client.create_thread_with_message(
                user_prompt=prompt,
                image_file_ids=uploaded_file_ids,
                standard_file_ids=standard_file_ids
            )
        
        # Run the assistant and get response
        response_text = await client.run_assistant(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        return {
            "success": True,
            "assessment": response_text,
            "thread_id": thread_id,
            "images_analyzed": len(uploaded_file_ids),
            "standards_used": len(standard_file_ids)
        }
        
    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "images_analyzed": len(uploaded_file_ids),
            "thread_id": thread_id  # Return thread_id even on failure for debugging
        }
    
    finally:
        # Clean up uploaded user files (not standards - those are reused)
        if uploaded_file_ids:
            logger.info(f"Cleaning up {len(uploaded_file_ids)} uploaded user files")
            client.delete_files_batch(uploaded_file_ids)
