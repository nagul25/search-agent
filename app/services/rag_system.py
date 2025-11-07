"""
RAG System Orchestrator
Integrates query analysis, document retrieval, and answer generation
"""

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
    
    def answer_question(self, question: str, top_k: int = 100, retrieve_all: bool = True) -> Dict[str, Any]:
        logging.info("Initiated:: Answering question using RAG System")

        """
        Answer a user question using RAG approach
        
        Args:
            question: User's natural language question
            top_k: Maximum number of documents to retrieve (default: 100)
            retrieve_all: If True, retrieves all relevant documents up to top_k limit
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        
        result = {
            "question": question,
            "answer": "",
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
            answer = self._generate_answer(question, context, documents)
            result["answer"] = answer
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
    
    def _generate_answer(self, question: str, context: str, documents: List[Dict]) -> str:
        """Generate answer using Azure AI Foundry GPT-5"""
        
        system_prompt = """You are an expert technology tools assistant for Experian. Your primary role is to answer questions about software tools and technology standards based solely on the provided context.

KNOWLEDGE BASE INFORMATION:
The knowledge base contains the "Experian technology standard list", which includes:
1. Software tools and their capabilities used within Experian organization
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

ANSWER GENERATION GUIDELINES:
1. STRICT CONTEXT USAGE: Answer ONLY using information from the provided context. Do not use external knowledge or assumptions.
2. ACCURACY: Cite specific tool names, manufacturers, versions, and TEB status when available in the context.
3. COMPLETENESS: Include relevant details from all fields including capabilities, sub-capabilities, descriptions (where full terms like "publish/subscribe" may appear), and meta tags when they help answer the question.
4. CLARITY: Structure your answer clearly, prioritizing the most relevant information first. When user uses abbreviations like "pub/sub", acknowledge their terminology while providing complete information.
5. HANDLING INCOMPLETE INFORMATION: If the context lacks sufficient information to fully answer the question, explicitly state what information is missing and provide partial answers based on available context.
6. RELEVANCE: Focus on tools and information that directly address the user's question, even if they used different terminology (abbreviations, synonyms). Avoid unnecessary details.
7. TEB STATUS: Always mention TEB approval status when discussing tool adoption or standards, as this is critical for organizational compliance."""

        user_prompt = f"""Context (Retrieved Technology Tools):

{context}

Question: {question}

Provide a comprehensive answer based on the context provided above."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Some models only support default temperature (1)
            # Remove temperature parameter for gpt-5 compatibility
            response = self.foundry_client.chat.completions.create(
                model=self.foundry_deployment,
                messages=messages
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

