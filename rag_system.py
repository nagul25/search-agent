"""
RAG System Orchestrator
Integrates query analysis, document retrieval, and answer generation
"""

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
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a user question using RAG approach
        
        Args:
            question: User's natural language question
            top_k: Number of documents to retrieve
            
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
                             "MetaTags", "Version"]
            )
            
            if "error" in search_results:
                result["answer"] = f"Error retrieving documents: {search_results['error']}"
                return result
            
            result["metadata"]["documents_retrieved"] = search_results.get("total_count", 0)
            documents = search_results.get("results", [])
            
            if not documents:
                result["answer"] = "No relevant documents found in the knowledge base to answer your question."
                return result
            
            # Step 3: Format documents as context
            context = self._format_documents_as_context(documents)
            
            # Step 4: Generate answer using Azure AI Foundry GPT-5
            print(f"Generating answer...")
            answer = self._generate_answer(question, context, documents)
            result["answer"] = answer
            result["sources"] = documents[:top_k]
            
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
            
            if doc.get('Description'):
                context_part += f"Description: {doc.get('Description')}\n"
            
            if doc.get('MetaTags'):
                context_part += f"Tags: {doc.get('MetaTags')}\n"
            
            if '@search.score' in doc:
                context_part += f"Relevance Score: {doc.get('@search.score', 'N/A')}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, documents: List[Dict]) -> str:
        """Generate answer using Azure AI Foundry GPT-5"""
        
        system_prompt = """You are a technology tools expert assistant. Your role is to answer questions about technology tools based on the context provided to you.

Guidelines:
- Answer based ONLY on the context provided
- Be accurate and concise
- If the context doesn't contain enough information to answer the question, say so
- Provide specific tool names and capabilities when relevant
- Include TEB approval status and manufacturer information when relevant"""

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

