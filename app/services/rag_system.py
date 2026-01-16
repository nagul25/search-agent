"""
RAG System Orchestrator
Integrates query analysis, document retrieval, and answer generation
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from openai import AzureOpenAI
from query_analyzer import QueryAnalyzer
from hybrid_search_client import HybridSearchClient
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.search_client = HybridSearchClient()
        
        # Azure AI Foundry configuration for GPT-5
        self.foundry_endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
        self.foundry_key = os.getenv("AZURE_AI_FOUNDRY_KEY")
        self.foundry_deployment = os.getenv("AZURE_AI_FOUNDRY_DEPLOYMENT", "gpt-5")
        self.foundry_api_version = os.getenv("AZURE_AI_FOUNDRY_API_VERSION", "2024-02-15-preview")
        
        self._initialize_foundry_client()
    
    def _initialize_foundry_client(self):
        """Initialize Azure AI Foundry client"""
        try:
            self.foundry_client = AzureOpenAI(
                api_key=self.foundry_key,
                api_version=self.foundry_api_version,
                azure_endpoint=self.foundry_endpoint
            )
        except Exception as e:
            print(f"Error initializing Azure AI Foundry client: {str(e)}")
            raise
    
    def answer_question(self, question: str, top_k: int = 50, retrieve_all: bool = True) -> Dict[str, Any]:
        logging.info("Initiated:: Answering question using RAG System")

        """
        Answer a user question using RAG approach
        
        Args:
            question: User's natural language question
            top_k: Maximum number of documents to retrieve (default: 50)
            retrieve_all: If True, retrieves all relevant documents up to top_k limit
            
        Returns:
            Dictionary with answer (containing ALL relevant tools with complete fields), sources, and metadata
        """
        
        result = {
            "question": question,
            "answer": "",
            "tools": [],
            "sources": [],
            "metadata": {
                "search_query": "",
                "filters": "",
                "intent": "",
                "documents_retrieved": 0
            }
        }
        
        try:
            # Step 1: Analyze question and extract search parameters
            print(f"\nAnalyzing question...")
            analysis = self.query_analyzer.analyze_question(question)
            result["metadata"]["search_query"] = analysis["search_query"]
            result["metadata"]["filters"] = analysis["filters"]
            result["metadata"]["intent"] = analysis["intent"]
            
            # Step 2: Retrieve relevant documents
            print(f"Searching for relevant documents...")
            search_results = self.search_client.hybrid_search(
                query=analysis["search_query"],
                filters=analysis["filters"] if analysis["filters"] else None,
                top=top_k,
                select_fields=["NameofTools", "Manufacturer", "TEBStatus", 
                             "Capabilities", "SubCapability", "Description", 
                             "MetaTags", "Version", "StandardsComments", 
                             "EANotes", "StandardCategory", "EAReferenceID", 
                             "MetaTagsDescription", "CapabilityManager"]
            )
            
            if "error" in search_results:
                result["answer"] = f"Error retrieving documents: {search_results['error']}"
                return result
            
            total_count = search_results.get("total_count", 0)
            documents = search_results.get("results", [])
            
            result["metadata"]["documents_retrieved"] = total_count
            
            print(f"Found {total_count} relevant documents, retrieved {len(documents)} documents")
            
            if not documents:
                result["answer"] = "No relevant documents found in the knowledge base to answer your question."
                return result
            
            # Step 3: Format documents as context
            context = self._format_documents_as_context(documents)
            
            # Step 4: Generate answer using Azure AI Foundry GPT-5
            print(f"Generating answer using {len(documents)} documents as context...")
            generated_response = self._generate_answer(question, context, documents)
            result["answer"] = generated_response.get("summary", "")
            result["tools"] = generated_response.get("tools", [])
            result["sources"] = documents
            
            return result
            
        except Exception as e:
            result["answer"] = f"Error processing question: {str(e)}"
            return result
    
    def _format_documents_as_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context for the LLM"""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_part = f"Document {i}:\n"
            context_part += f"Tool Name: {doc.get('NameofTools', 'N/A')}\n"
            context_part += f"Manufacturer: {doc.get('Manufacturer', 'N/A')}\n"
            context_part += f"TEB Status: {doc.get('TEBStatus', 'N/A')}\n"
            context_part += f"Capability: {doc.get('Capabilities', 'N/A')}\n"
            context_part += f"Sub-Capability: {doc.get('SubCapability', 'N/A')}\n"
            context_part += f"Version: {doc.get('Version', 'N/A')}\n"
            context_part += f"Standard Category: {doc.get('StandardCategory', 'N/A')}\n"
            context_part += f"EA Reference ID: {doc.get('EAReferenceID', 'N/A')}\n"
            context_part += f"Capability Manager: {doc.get('CapabilityManager', 'N/A')}\n"
            
            if doc.get('Description'):
                context_part += f"Description: {doc.get('Description')}\n"
            
            if doc.get('StandardsComments'):
                context_part += f"Standards Comments: {doc.get('StandardsComments')}\n"
            
            if doc.get('EANotes'):
                context_part += f"EA Notes: {doc.get('EANotes')}\n"
            
            if doc.get('MetaTags'):
                context_part += f"Meta Tags: {doc.get('MetaTags')}\n"
            
            if doc.get('MetaTagsDescription'):
                context_part += f"Meta Tags Description: {doc.get('MetaTagsDescription')}\n"
            
            if '@search.score' in doc:
                context_part += f"Relevance Score: {doc.get('@search.score', 'N/A')}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, documents: List[Dict]) -> Dict[str, Any]:
        """
        Generate answer using Azure AI Foundry GPT-5
        Returns structured JSON with summary and tools array for frontend display.
        
        Returns:
            Dictionary containing:
                - summary: Text summary with knowledge source indication
                - tools: Array of tool objects with structured attributes
        """
        
        system_prompt = """You are an expert technology tools assistant for Mahaaya. Your primary role is to answer questions about software tools and technology standards.

KNOWLEDGE BASE INFORMATION:
The knowledge base contains the "Mahaaya technology standard list", which includes:
1. Software tools and their capabilities used within Mahaaya organization
2. Tool categorization by:
   - Capabilities 
   - Sub-capabilities
   - Tool names (NameofTools)
   - Manufacturers
   - Versions
   - Meta tags (additional functionality descriptions)
3. Technology Evaluation Board (TEB) approval status for each tool
4. Standard consideration and approval processes managed through the TEB process

AVAILABLE DATA FIELDS:
Each tool document contains all columns from the CSV: NameofTools, Manufacturer, TEBStatus, Capabilities, SubCapability, Description, MetaTags, Version, StandardsComments, EANotes, StandardCategory, EAReferenceID, MetaTagsDescription, and CapabilityManager. Use all available fields to provide accurate and detailed answers.

SEMANTIC UNDERSTANDING:
1. ABBREVIATIONS AND SHORTHAND: Recognize and interpret common technical abbreviations and shorthand when matching user questions to context. Examples:
   - "pub/sub" or "pubsub" = "publish/subscribe" or "publishing and subscribing"
   - "devops" or "DevOps" = "DevOps" (case variations)
   - "I&A" or "IAM" = "Identity & Access Management"
   - "CI/CD" = "Continuous Integration/Continuous Deployment"
2. SYNONYMS AND VARIATIONS: Understand semantic equivalents and terminology variations:
   - "authentication" = "auth" = "identity verification"
   - "messaging" = "event streaming" = "message queue" (when contextually relevant)
   - "analytics" = "data analytics" = "business intelligence" (where applicable)
3. COMPREHENSIVE FIELD SEARCH: When matching user terminology to tools, check ALL available fields in the context including:
   - NameofTools (tool name)
   - Capabilities 
   - SubCapability 
   - Description (detailed tool descriptions - CRITICAL: full terms like "publish/subscribe" often appear here)
   - MetaTags (additional functionality tags)
   - Manufacturer
   - Version
4. TERMINOLOGY MAPPING: Map user's terminology to knowledge base terminology across ALL fields. If user asks about "pub/sub tools", look for tools where ANY field (especially Description, Capabilities, SubCapability, or MetaTags) contains terms related to:
   - "publish/subscribe" or "publishing and subscribing" (full term may be in Description)
   - "messaging"
   - "event streaming"
   - "message queue"
   - "pub/sub" (if explicitly mentioned)
5. CONTEXT MATCHING: Match user questions to tools in context even if exact terminology differs, as long as the semantic meaning aligns. Example: When user asks "pub/sub tools", match tools where "publish/subscribe" appears in Description, capabilities, or any other field, even if the exact abbreviation "pub/sub" is not present in the context.

HANDLING TOOLS NOT IN CONTEXT (IMPORTANT):
When the user asks about a specific tool that is NOT present in the provided context:

1. PRIORITY ORDER:
   - FIRST: Provide information about the tool the user specifically asked about (use external knowledge)
   - SECOND: Include the closest related/similar tools FROM THE CONTEXT as Mahaaya-approved alternatives

2. FINDING RELATED TOOLS:
   When the requested tool is not in context, identify related tools by matching:
   - Same category/capability (e.g., if user asks about MongoDB, include document databases from context)
   - Similar functionality (e.g., if user asks about Jenkins, include CI/CD tools from context)
   - Same use case (e.g., if user asks about Slack, include communication/collaboration tools from context)
   - Same technology domain (e.g., if user asks about Redis, include caching or in-memory data tools from context)

3. RESPONSE STRUCTURE FOR MISSING TOOLS:
   - Clearly answer the user's question about the requested tool first
   - Then introduce related alternatives with: "While [requested tool] is not in the Mahaaya standards list, the following related tools are available:"
   - Explain why these alternatives are relevant (similar capability, use case, etc.)

4. TOOLS ARRAY BEHAVIOR:
   - If the requested tool IS in context: include it in the tools array
   - If the requested tool is NOT in context: include the closest related tools from context in the tools array
   - Mark external knowledge tools clearly in the summary, but only include context-based tools in the tools array

5. EXAMPLES:
   - User asks about "CouchDB" (not in context) -> Provide CouchDB info, then include document DBs or NoSQL tools from context
   - User asks about "Terraform" (not in context) -> Provide Terraform info, then include IaC or DevOps tools from context
   - User asks about "Datadog" (not in context) -> Provide Datadog info, then include monitoring/observability tools from context

RESPONSE FORMAT:
You MUST respond with a valid JSON object containing exactly two fields:
1. "summary": A well-formatted text summary answering the user's question (see SUMMARY FORMATTING below)
2. "tools": An array of tool objects that are relevant to the question

SUMMARY FORMATTING (CRITICAL FOR READABILITY):
The summary field MUST be formatted with proper structure for easy reading:

1. START with the knowledge source label on its own line
2. Use DOUBLE LINE BREAKS (\\n\\n) between paragraphs and sections
3. Use MARKDOWN formatting for structure:
   - Use **bold** for emphasis and section headers
   - Use bullet points (- item) for lists
   - Use numbered lists (1. item) for sequential information
4. Organize content into clear sections when appropriate:
   - Overview/Introduction
   - Key Points or Comparison (if comparing tools)
   - Recommendations (if applicable)
   - Notes or Caveats (if applicable)
5. Keep paragraphs SHORT (2-4 sentences max)
6. Use line breaks to separate distinct ideas

KNOWLEDGE SOURCE INDICATION (first line of summary):
- "[KNOWLEDGE SOURCE: Context Only]" - if answering solely from provided context
- "[KNOWLEDGE SOURCE: External Knowledge]" - if answering primarily from external knowledge
- "[KNOWLEDGE SOURCE: Context + External Knowledge]" - if combining both sources

TOOL OBJECT STRUCTURE:
Each tool in the "tools" array MUST have this exact structure:
{
    "name": "Tool name from NameofTools field",
    "manufacturer": "Manufacturer name",
    "version": "Version string or null if not available",
    "tebStatus": "TEB approval status",
    "capability": "Primary capability",
    "subCapability": "Sub-capability or null if not available",
    "description": "Tool description or null if not available",
    "standardCategory": "Standard category or null if not available",
    "eaReferenceId": "EA Reference ID or null if not available",
    "capabilityManager": "Capability manager or null if not available",
    "metaTags": "Meta tags or null if not available",
    "standardsComments": "Standards comments or null if not available",
    "eaNotes": "EA notes or null if not available"
}

IMPORTANT RULES:
1. Only include tools that are directly relevant to answering the user's question
2. Do NOT include all tools from context - filter to only the most relevant ones
3. Use null for any field that is not available or marked as "N/A" in the context
4. Ensure the JSON is valid and properly formatted
5. Do NOT include any text outside the JSON object
6. Prioritize answering the user's specific question FIRST, then provide Mahaaya alternatives
7. NEVER write the summary as a single long paragraph - always use proper formatting
8. When requested tool is NOT in context, include closest related tools from context as alternatives
9. The tools array should ONLY contain tools from context (not external knowledge tools)

EXAMPLE 1 - Tool found in context:
{
    "summary": "[KNOWLEDGE SOURCE: Context Only]\\n\\n**Overview**\\n\\nBased on the Mahaaya technology standards, there are 2 tools available for messaging capabilities.\\n\\n**Available Tools**\\n\\n- **Apache Kafka** - A distributed event streaming platform (TEB Status: Approved)\\n- **RabbitMQ** - A message broker for async communication (TEB Status: Under Review)\\n\\n**Recommendation**\\n\\nFor production use, Apache Kafka is recommended as it has full TEB approval.",
    "tools": [
        {
            "name": "Apache Kafka",
            "manufacturer": "Apache Software Foundation",
            "version": "3.0",
            "tebStatus": "Approved",
            "capability": "Data Integration",
            "subCapability": "Event Streaming",
            "description": "Distributed event streaming platform",
            "standardCategory": "Standard",
            "eaReferenceId": "EA-001",
            "capabilityManager": "John Doe",
            "metaTags": "messaging, streaming",
            "standardsComments": null,
            "eaNotes": null
        }
    ]
}

EXAMPLE 2 - Tool NOT in context (include related alternatives):
{
    "summary": "[KNOWLEDGE SOURCE: Context + External Knowledge]\\n\\n**About MongoDB**\\n\\nMongoDB is a popular open-source NoSQL document database developed by MongoDB Inc. It stores data in flexible, JSON-like documents and is widely used for modern web applications.\\n\\n**Key Features**\\n\\n- Document-oriented storage with dynamic schemas\\n- Horizontal scaling with sharding\\n- Rich query language and indexing\\n\\n**Mahaaya Alternatives**\\n\\nWhile MongoDB is not currently in the Mahaaya technology standards list, the following related database tools are approved:\\n\\n- **Oracle Database** - Enterprise relational database (TEB Status: Approved)\\n- **PostgreSQL** - Open-source relational database (TEB Status: Approved)\\n\\n**Note**\\n\\nBefore using MongoDB, please consult with the TEB for approval or consider the approved alternatives listed above.",
    "tools": [
        {
            "name": "Oracle Database",
            "manufacturer": "Oracle Corporation",
            "version": "19c",
            "tebStatus": "Approved",
            "capability": "Data Management",
            "subCapability": "Relational Database",
            "description": "Enterprise-grade relational database management system",
            "standardCategory": "Standard",
            "eaReferenceId": "EA-DB-001",
            "capabilityManager": "Jane Smith",
            "metaTags": "database, sql, enterprise",
            "standardsComments": null,
            "eaNotes": null
        }
    ]
}"""

        user_prompt = f"""Context (Retrieved Technology Tools):

{context}

Question: {question}

Respond with a JSON object containing:
1. A well-formatted "summary" with proper line breaks, sections, and markdown formatting for readability
2. A "tools" array with relevant tool objects

Remember: Format the summary with clear structure - use double line breaks between sections, bullet points for lists, and bold for headers. Never write a wall of text."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Request JSON response format
            response = self.foundry_client.chat.completions.create(
                model=self.foundry_deployment,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            try:
                parsed_response = json.loads(response_text)
                return {
                    "summary": parsed_response.get("summary", ""),
                    "tools": parsed_response.get("tools", [])
                }
            except json.JSONDecodeError as parse_error:
                logging.warning(f"Failed to parse JSON response: {parse_error}")
                # Fallback: return the raw text as summary with empty tools
                return {
                    "summary": response_text,
                    "tools": []
                }
            
        except Exception as e:
            return {
                "summary": f"Error generating answer: {str(e)}",
                "tools": []
            }

