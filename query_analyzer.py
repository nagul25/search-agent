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
   - Expand abbreviations to include both abbreviated and full forms (e.g., "pub/sub" → include "pub sub", "publish subscribe", "publish/subscribe")
   - This ensures matches in Description fields where full terms often appear
2. OData filter expressions for Azure AI Search
   - Only include filters for values that are EXPLICITLY mentioned in the question
   - Do not infer or assume filter values that are not present in the question
   - If a field value is not mentioned, do not include it in the filters
3. Intent classification

Available filter fields (extract these values ONLY if present in the question):
- TEBStatus: Can only be 'TEB Approved' or 'TEB Not Approved'. Only include if the question mentions TEB approval status.
- Manufacturer: Can be any manufacturer name mentioned in the question (e.g., 'Google', 'Microsoft', 'Amazon', 'IBM', 'Oracle', etc.). Extract the exact manufacturer name ONLY if mentioned in the question.
- Capabilities: Can be any capability mentioned in the question (e.g., 'Identity & Access Mgmt', 'DevOps', 'Analytics', 'Data Management', 'Security', etc.). Extract the exact capability name ONLY if mentioned in the question.
- SubCapability: Can be any sub-capability mentioned in the question. Extract the exact sub-capability name ONLY if mentioned in the question.

Filter Operators:
- eq (equals): TEBStatus eq 'TEB Approved'
- ne (not equals): TEBStatus ne 'TEB Not Approved'
- or: Manufacturer eq 'Google' or Manufacturer eq 'Microsoft'
- and: TEBStatus eq 'TEB Approved' and Capabilities eq 'DevOps'

Examples:
Question: "What TEB approved authentication tools are available?"
- search_query: "authentication tools identity access"
- filters: "TEBStatus eq 'TEB Approved' and Capabilities eq 'Identity & Access Mgmt'"
- intent: "Filter by TEB Approved authentication tools"

Question: "Show me Google's pub/sub messaging tools"
- search_query: "pub sub publish subscribe publish/subscribe messaging event streaming"
- filters: "Manufacturer eq 'Google'"
- intent: "Google pub/sub messaging tools"

Question: "Which DevOps tools are not TEB approved?"
- search_query: "devops ci/cd pipeline automation"
- filters: "TEBStatus eq 'TEB Not Approved' and Capabilities eq 'DevOps'"
- intent: "DevOps tools not TEB approved"

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

