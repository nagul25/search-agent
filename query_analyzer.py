"""
Query Analyzer for RAG System
Uses LLM to analyze user questions and extract search parameters
"""

import os
import json
from typing import Dict, Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

class QueryAnalyzer:
    def __init__(self):
        # Azure AI Foundry configuration (primary for query analysis)
        foundry_endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
        foundry_key = os.getenv("AZURE_AI_FOUNDRY_KEY")
        foundry_api_version = os.getenv("AZURE_AI_FOUNDRY_API_VERSION", "2024-02-15-preview")
        self.analysis_model = os.getenv("ANALYSIS_MODEL", "gpt-5-mini")
        
        # Fallback to OpenAI if Foundry not configured
        if not foundry_endpoint or not foundry_key:
            print("Warning: Azure AI Foundry not configured. Falling back to OpenAI.")
            foundry_endpoint = os.getenv("OPENAI_ENDPOINT")
            foundry_key = os.getenv("OPENAI_API_KEY")
            foundry_api_version = os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
        
        self.openai_client = AzureOpenAI(
            api_key=foundry_key,
            api_version=foundry_api_version,
            azure_endpoint=foundry_endpoint
        )
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze user question and extract search parameters
        
        Args:
            question: User's natural language question
            
        Returns:
            Dictionary with search_query, filters, and intent
        """
        
        analysis_prompt = """You are an expert at analyzing questions about technology tools and extracting search parameters.

Your task is to analyze the user's question and extract:
1. Search keywords for hybrid search (extract key terms for semantic/vector search)
2. OData filter expressions for Azure AI Search
3. Intent classification

Available filter fields and values:
- TEBStatus: 'TEB Approved', 'TEB Not Approved', 'Under Review', 'Deprecated'
- Manufacturer: 'Google', 'Microsoft', 'Amazon', 'IBM', 'Oracle', etc.
- Capabilities: 'Identity & Access Mgmt', 'DevOps', 'Analytics', 'Data Management', 'Security', etc.
- SubCapability: Various sub-categories based on capability

Filter Operators:
- eq (equals): TEBStatus eq 'TEB Approved'
- ne (not equals): TEBStatus ne 'Deprecated'
- or: Manufacturer eq 'Google' or Manufacturer eq 'Microsoft'
- and: TEBStatus eq 'TEB Approved' and Capabilities eq 'DevOps'

Examples:
Question: "What TEB approved authentication tools are available?"
- search_query: "authentication tools identity access"
- filters: "TEBStatus eq 'TEB Approved' and Capabilities eq 'Identity & Access Mgmt'"
- intent: "Filter by TEB Approved authentication tools"

Question: "Show me Google's pub/sub messaging tools"
- search_query: "pub sub messaging event streaming"
- filters: "Manufacturer eq 'Google'"
- intent: "Google pub/sub messaging tools"

Question: "Which DevOps tools are under review?"
- search_query: "devops ci/cd pipeline automation"
- filters: "TEBStatus eq 'Under Review' and Capabilities eq 'DevOps'"
- intent: "DevOps tools under review"

Question: "What security tools can I use?"
- search_query: "security compliance governance"
- filters: ""
- intent: "General security tools query"

Question: "List all available tools"
- search_query: "*"
- filters: ""
- intent: "List all tools"

Question: {question}

Return ONLY a JSON object with this exact format:
{{"search_query": "...", "filters": "...", "intent": "..."}}

Do not include any explanations or additional text."""

        try:
            messages = [
                {"role": "system", "content": analysis_prompt.format(question=question)},
                {"role": "user", "content": f"Analyze this question: {question}"}
            ]
            
            # Some models only support default temperature (1)
            # Remove temperature parameter for gpt-5-mini compatibility
            response = self.openai_client.chat.completions.create(
                model=self.analysis_model,
                messages=messages
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            result = json.loads(content)
            
            return {
                "search_query": result.get("search_query", question),
                "filters": result.get("filters", ""),
                "intent": result.get("intent", "General query")
            }
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response content: {content}")
            return {
                "search_query": question,
                "filters": "",
                "intent": "General query"
            }
        except Exception as e:
            print(f"Error analyzing question: {str(e)}")
            return {
                "search_query": question,
                "filters": "",
                "intent": "Error in analysis"
            }

