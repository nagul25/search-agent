"""
Azure AI Search Hybrid Search Client
Supports vector search, keyword search, semantic search, and filtering
"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HybridSearchClient:
    def __init__(self):
        # Azure AI Search configuration
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_KEY")
        self.search_index_name = "technology-tools-index"
        
        # Azure OpenAI configuration for embeddings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.openai_api_version = os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Azure clients"""
        try:
            # Azure AI Search client
            credential = AzureKeyCredential(self.search_key)
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.search_index_name,
                credential=credential
            )
            
            # Azure OpenAI client for embeddings
            self.openai_client = AzureOpenAI(
                api_key=self.openai_api_key,
                api_version=self.openai_api_version,
                azure_endpoint=self.openai_endpoint
            )
            
            print("Hybrid Search Client initialized successfully")
            
        except Exception as e:
            print(f"Error initializing search client: {str(e)}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using Azure OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            return []
    
    def keyword_search(self, 
                      query: str, 
                      filters: Optional[str] = None,
                      top: int = 100,
                      select_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform keyword search with optional filtering
        
        Args:
            query: Search query text
            filters: OData filter expression (e.g., "teb_status eq 'TEB Approved'")
            top: Number of results to return (default: 100 to retrieve all relevant documents)
            select_fields: Fields to return in results
        """
        try:
            search_params = {
                "search_text": query,
                "top": top,
                "include_total_count": True
            }
            
            if filters:
                search_params["filter"] = filters
            
            if select_fields:
                search_params["select"] = select_fields
            
            results = self.search_client.search(**search_params)
            
            return {
                "query_type": "keyword",
                "query": query,
                "filters": filters,
                "total_count": results.get_count(),
                "results": [dict(result) for result in results]
            }
            
        except Exception as e:
            print(f"Error in keyword search: {str(e)}")
            return {"error": str(e)}
    
    def vector_search(self, 
                     query: str, 
                     filters: Optional[str] = None,
                     top: int = 100,
                     select_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform vector search with optional filtering
        
        Args:
            query: Search query text
            filters: OData filter expression
            top: Number of results to return (default: 100 to retrieve all relevant documents)
            select_fields: Fields to return in results
        """
        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)
            if not query_embedding:
                return {"error": "Failed to create embedding for query"}
            
            # Create vectorized query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top,
                fields="content_vector"
            )
            
            search_params = {
                "vector_queries": [vector_query],
                "top": top,
                "include_total_count": True
            }
            
            if filters:
                search_params["filter"] = filters
            
            if select_fields:
                search_params["select"] = select_fields
            
            results = self.search_client.search(**search_params)
            
            return {
                "query_type": "vector",
                "query": query,
                "filters": filters,
                "total_count": results.get_count(),
                "results": [dict(result) for result in results]
            }
            
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            return {"error": str(e)}
    
    def hybrid_search(self, 
                     query: str, 
                     filters: Optional[str] = None,
                     top: int = 100,
                     select_fields: Optional[List[str]] = None,
                     semantic_configuration_name: str = "default-semantic-config") -> Dict[str, Any]:
        """
        Perform hybrid search (vector + keyword + semantic) with optional filtering
        
        Args:
            query: Search query text
            filters: OData filter expression
            top: Number of results to return (default: 100 to retrieve all relevant documents)
            select_fields: Fields to return in results
            semantic_configuration_name: Semantic search configuration name
        """
        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)
            if not query_embedding:
                return {"error": "Failed to create embedding for query"}
            
            # Create vectorized query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top,
                fields="content_vector"
            )
            
            search_params = {
                "search_text": query,
                "vector_queries": [vector_query],
                "top": top,
                "include_total_count": True,
                "query_type": QueryType.SEMANTIC,
                "semantic_configuration_name": semantic_configuration_name,
                "query_caption": QueryCaptionType.EXTRACTIVE,
                "query_answer": QueryAnswerType.EXTRACTIVE
            }
            
            if filters:
                search_params["filter"] = filters
            
            if select_fields:
                search_params["select"] = select_fields
            
            results = self.search_client.search(**search_params)
            
            return {
                "query_type": "hybrid",
                "query": query,
                "filters": filters,
                "total_count": results.get_count(),
                "results": [dict(result) for result in results]
            }
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return {"error": str(e)}
    
    def filter_search(self, 
                     filters: str,
                     top: int = 100,
                     select_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform filtered search without text query
        
        Args:
            filters: OData filter expression
            top: Number of results to return (default: 100 to retrieve all relevant documents)
            select_fields: Fields to return in results
        """
        try:
            search_params = {
                "search_text": "*",
                "filter": filters,
                "top": top,
                "include_total_count": True
            }
            
            if select_fields:
                search_params["select"] = select_fields
            
            results = self.search_client.search(**search_params)
            
            return {
                "query_type": "filter",
                "query": "*",
                "filters": filters,
                "total_count": results.get_count(),
                "results": [dict(result) for result in results]
            }
            
        except Exception as e:
            print(f"Error in filter search: {str(e)}")
            return {"error": str(e)}
    
    def get_facet_counts(self, 
                        search_text: str = "*",
                        facets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get facet counts for specified fields
        
        Args:
            search_text: Search text (default: "*" for all)
            facets: List of fields to facet on
        """
        try:
            if not facets:
                facets = ["TEBStatus", "Manufacturer", "Capabilities", "SubCapability"]
            
            search_params = {
                "search_text": search_text,
                "facets": facets,
                "top": 0,
                "include_total_count": True
            }
            
            results = self.search_client.search(**search_params)
            
            return {
                "query_type": "facets",
                "search_text": search_text,
                "facets": facets,
                "total_count": results.get_count(),
                "facet_counts": dict(results.get_facets())
            }
            
        except Exception as e:
            print(f"Error getting facet counts: {str(e)}")
            return {"error": str(e)}
    
    def search_with_examples(self):
        """Demonstrate various search capabilities with examples"""
        print("Azure AI Search Examples")
        print("=" * 50)
        
        # Example 1: Filter by TEB Status
        print("\n1. Filter by TEB Status - TEB Approved tools:")
        result = self.filter_search("TEBStatus eq 'TEB Approved'")
        print(f"Found {result.get('total_count', 0)} TEB Approved tools")
        for i, doc in enumerate(result.get('results', [])[:3]):
            print(f"  {i+1}. {doc.get('NameofTools')} - {doc.get('Manufacturer')}")
        
        # Example 2: Filter by Manufacturer
        print("\n2. Filter by Manufacturer - Google tools:")
        result = self.filter_search("Manufacturer eq 'Google'")
        print(f"Found {result.get('total_count', 0)} Google tools")
        for i, doc in enumerate(result.get('results', [])[:3]):
            print(f"  {i+1}. {doc.get('NameofTools')} - {doc.get('TEBStatus')}")
        
        # Example 3: Hybrid search for authentication tools
        print("\n3. Hybrid search for authentication tools:")
        result = self.hybrid_search("authentication tools")
        print(f"Found {result.get('total_count', 0)} authentication-related tools")
        for i, doc in enumerate(result.get('results', [])[:3]):
            print(f"  {i+1}. {doc.get('NameofTools')} - {doc.get('Capabilities')}")
        
        # Example 4: Hybrid search with filter - TEB Approved pub/sub tools
        print("\n4. Hybrid search with filter - TEB Approved pub/sub tools:")
        result = self.hybrid_search(
            "pub sub messaging", 
            filters="TEBStatus eq 'TEB Approved'"
        )
        print(f"Found {result.get('total_count', 0)} TEB Approved pub/sub tools")
        for i, doc in enumerate(result.get('results', [])[:3]):
            print(f"  {i+1}. {doc.get('NameofTools')} - {doc.get('Manufacturer')}")
        
        # Example 5: Keyword search for specific capability
        print("\n5. Keyword search for DevOps tools:")
        result = self.keyword_search("DevOps")
        print(f"Found {result.get('total_count', 0)} DevOps tools")
        for i, doc in enumerate(result.get('results', [])[:3]):
            print(f"  {i+1}. {doc.get('NameofTools')} - {doc.get('SubCapability')}")
        
        # Example 6: Get facet counts
        print("\n6. Facet counts for TEB Status:")
        result = self.get_facet_counts(facets=["TEBStatus"])
        facet_counts = result.get('facet_counts', {}).get('TEBStatus', [])
        for facet in facet_counts:
            print(f"  {facet['value']}: {facet['count']}")

def main():
    """Main function to demonstrate search capabilities"""
    try:
        # Initialize search client
        search_client = HybridSearchClient()
        
        # Run examples
        search_client.search_with_examples()
        
    except Exception as e:
        print(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()

