"""
Azure AI Search Examples
Demonstrates all search capabilities including filtering, hybrid search, and semantic search
"""

import os
import json
from typing import Dict, Any, List
from hybrid_search_client import HybridSearchClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SearchExamples:
    def __init__(self):
        self.search_client = HybridSearchClient()
    
    def print_results(self, result: Dict[str, Any], max_results: int = 3):
        """Print search results in a formatted way"""
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"Query Type: {result.get('query_type', 'unknown')}")
        print(f"Query: {result.get('query', 'N/A')}")
        print(f"Filters: {result.get('filters', 'None')}")
        print(f"Total Results: {result.get('total_count', 0)}")
        print("-" * 50)
        
        results = result.get('results', [])
        for i, doc in enumerate(results[:max_results]):
            print(f"{i+1}. {doc.get('NameofTools', 'N/A')}")
            print(f"   Manufacturer: {doc.get('Manufacturer', 'N/A')}")
            print(f"   TEB Status: {doc.get('TEBStatus', 'N/A')}")
            print(f"   Capability: {doc.get('Capabilities', 'N/A')}")
            print(f"   Sub-capability: {doc.get('SubCapability', 'N/A')}")
            if doc.get('Description'):
                desc = doc['Description'][:100] + "..." if len(doc['Description']) > 100 else doc['Description']
                print(f"   Description: {desc}")
            print()
    
    def example_1_filter_by_teb_status(self):
        """Example 1: Filter by TEB Status"""
        print("EXAMPLE 1: Filter by TEB Status")
        print("=" * 60)
        
        # TEB Approved tools
        print("\n1. TEB Approved tools:")
        result = self.search_client.filter_search("TEBStatus eq 'TEB Approved'")
        self.print_results(result)
        
        # TEB Not Approved tools
        print("\n2. TEB Not Approved tools:")
        result = self.search_client.filter_search("TEBStatus eq 'TEB Not Approved'")
        self.print_results(result)
        
        # Under Review tools
        print("\n3. Under Review tools:")
        result = self.search_client.filter_search("TEBStatus eq 'Under Review'")
        self.print_results(result)
    
    def example_2_filter_by_manufacturer(self):
        """Example 2: Filter by Manufacturer"""
        print("\nEXAMPLE 2: Filter by Manufacturer")
        print("=" * 60)
        
        # Google tools
        print("\n1. Google tools:")
        result = self.search_client.filter_search("Manufacturer eq 'Google'")
        self.print_results(result)
        
        # Microsoft tools
        print("\n2. Microsoft tools:")
        result = self.search_client.filter_search("Manufacturer eq 'Microsoft'")
        self.print_results(result)
        
        # Amazon tools
        print("\n3. Amazon tools:")
        result = self.search_client.filter_search("Manufacturer eq 'Amazon'")
        self.print_results(result)
    
    def example_3_filter_by_capability(self):
        """Example 3: Filter by Capability"""
        print("\nEXAMPLE 3: Filter by Capability")
        print("=" * 60)
        
        # Identity & Access Management tools
        print("\n1. Identity & Access Management tools:")
        result = self.search_client.filter_search("Capabilities eq 'Identity & Access Mgmt'")
        self.print_results(result)
        
        # DevOps tools
        print("\n2. DevOps tools:")
        result = self.search_client.filter_search("Capabilities eq 'DevOps'")
        self.print_results(result)
        
        # Analytics tools
        print("\n3. Analytics tools:")
        result = self.search_client.filter_search("Capabilities eq 'Analytics'")
        self.print_results(result)
    
    def example_4_hybrid_search(self):
        """Example 4: Hybrid Search (Vector + Keyword + Semantic)"""
        print("\nEXAMPLE 4: Hybrid Search")
        print("=" * 60)
        
        # Authentication tools
        print("\n1. Authentication tools:")
        result = self.search_client.hybrid_search("authentication tools")
        self.print_results(result)
        
        # Pub/Sub messaging tools
        print("\n2. Pub/Sub messaging tools:")
        result = self.search_client.hybrid_search("pub sub messaging")
        self.print_results(result)
        
        # Monitoring and observability tools
        print("\n3. Monitoring and observability tools:")
        result = self.search_client.hybrid_search("monitoring observability")
        self.print_results(result)
        
        # Data engineering tools
        print("\n4. Data engineering tools:")
        result = self.search_client.hybrid_search("data engineering ETL")
        self.print_results(result)
    
    def example_5_hybrid_search_with_filters(self):
        """Example 5: Hybrid Search with Filters"""
        print("\nEXAMPLE 5: Hybrid Search with Filters")
        print("=" * 60)
        
        # TEB Approved authentication tools
        print("\n1. TEB Approved authentication tools:")
        result = self.search_client.hybrid_search(
            "authentication tools",
            filters="TEBStatus eq 'TEB Approved'"
        )
        self.print_results(result)
        
        # Google pub/sub tools
        print("\n2. Google pub/sub tools:")
        result = self.search_client.hybrid_search(
            "pub sub messaging",
            filters="Manufacturer eq 'Google'"
        )
        self.print_results(result)
        
        # TEB Approved DevOps tools
        print("\n3. TEB Approved DevOps tools:")
        result = self.search_client.hybrid_search(
            "DevOps CI/CD",
            filters="TEBStatus eq 'TEB Approved' and Capabilities eq 'DevOps'"
        )
        self.print_results(result)
        
        # Under Review analytics tools
        print("\n4. Under Review analytics tools:")
        result = self.search_client.hybrid_search(
            "analytics data",
            filters="TEBStatus eq 'Under Review' and Capabilities eq 'Analytics'"
        )
        self.print_results(result)
    
    def example_6_keyword_search(self):
        """Example 6: Keyword Search"""
        print("\nEXAMPLE 6: Keyword Search")
        print("=" * 60)
        
        # Search for specific tools
        print("\n1. Tools containing 'Kubernetes':")
        result = self.search_client.keyword_search("Kubernetes")
        self.print_results(result)
        
        # Search for specific capabilities
        print("\n2. Tools for 'backup and recovery':")
        result = self.search_client.keyword_search("backup recovery")
        self.print_results(result)
        
        # Search for specific technologies
        print("\n3. Tools for 'machine learning':")
        result = self.search_client.keyword_search("machine learning ML")
        self.print_results(result)
    
    def example_7_vector_search(self):
        """Example 7: Vector Search"""
        print("\nEXAMPLE 7: Vector Search")
        print("=" * 60)
        
        # Semantic search for security tools
        print("\n1. Security and compliance tools:")
        result = self.search_client.vector_search("security compliance governance")
        self.print_results(result)
        
        # Semantic search for cloud platforms
        print("\n2. Cloud platform tools:")
        result = self.search_client.vector_search("cloud platform infrastructure")
        self.print_results(result)
        
        # Semantic search for data processing
        print("\n3. Data processing and analytics:")
        result = self.search_client.vector_search("data processing analytics insights")
        self.print_results(result)
    
    def example_8_complex_filters(self):
        """Example 8: Complex Filter Combinations"""
        print("\nEXAMPLE 8: Complex Filter Combinations")
        print("=" * 60)
        
        # Multiple manufacturer filter
        print("\n1. Tools from Google OR Microsoft:")
        result = self.search_client.filter_search(
            "Manufacturer eq 'Google' or Manufacturer eq 'Microsoft'"
        )
        self.print_results(result)
        
        # Status and capability combination
        print("\n2. TEB Approved Identity & Access Management tools:")
        result = self.search_client.filter_search(
            "TEBStatus eq 'TEB Approved' and Capabilities eq 'Identity & Access Mgmt'"
        )
        self.print_results(result)
        
        # Exclude deprecated tools
        print("\n3. Non-deprecated DevOps tools:")
        result = self.search_client.filter_search(
            "Capabilities eq 'DevOps' and TEBStatus ne 'Deprecated'"
        )
        self.print_results(result)
        
        # Version filtering (if version field has numeric values)
        print("\n4. Tools with version 3.x or higher:")
        result = self.search_client.filter_search(
            "Version ge '3.0.0'"
        )
        self.print_results(result)
    
    def example_9_facet_analysis(self):
        """Example 9: Facet Analysis"""
        print("\nEXAMPLE 9: Facet Analysis")
        print("=" * 60)
        
        # Get facet counts for TEB Status
        print("\n1. TEB Status distribution:")
        result = self.search_client.get_facet_counts(facets=["TEBStatus"])
        facet_counts = result.get('facet_counts', {}).get('TEBStatus', [])
        for facet in facet_counts:
            print(f"   {facet['value']}: {facet['count']} tools")
        
        # Get facet counts for Manufacturers
        print("\n2. Top Manufacturers:")
        result = self.search_client.get_facet_counts(facets=["Manufacturer"])
        facet_counts = result.get('facet_counts', {}).get('Manufacturer', [])
        for facet in facet_counts[:10]:  # Top 10
            print(f"   {facet['value']}: {facet['count']} tools")
        
        # Get facet counts for Capabilities
        print("\n3. Capability distribution:")
        result = self.search_client.get_facet_counts(facets=["Capabilities"])
        facet_counts = result.get('facet_counts', {}).get('Capabilities', [])
        for facet in facet_counts:
            print(f"   {facet['value']}: {facet['count']} tools")
    
    def example_10_advanced_queries(self):
        """Example 10: Advanced Query Examples"""
        print("\nEXAMPLE 10: Advanced Query Examples")
        print("=" * 60)
        
        # Search with specific field selection
        print("\n1. TEB Approved tools (selected fields only):")
        result = self.search_client.filter_search(
            "TEBStatus eq 'TEB Approved'",
            select_fields=["NameofTools", "Manufacturer", "TEBStatus", "Capabilities"]
        )
        self.print_results(result)
        
        # Hybrid search with multiple filters
        print("\n2. Google or Microsoft authentication tools:")
        result = self.search_client.hybrid_search(
            "authentication identity access",
            filters="(Manufacturer eq 'Google' or Manufacturer eq 'Microsoft') and Capabilities eq 'Identity & Access Mgmt'"
        )
        self.print_results(result)
        
        # Search with wildcards
        print("\n3. Tools with 'Azure' in the name:")
        result = self.search_client.keyword_search("Azure*")
        self.print_results(result)
    
    def run_all_examples(self):
        """Run all search examples"""
        print("Azure AI Search - Complete Examples Suite")
        print("=" * 80)
        
        try:
            self.example_1_filter_by_teb_status()
            self.example_2_filter_by_manufacturer()
            self.example_3_filter_by_capability()
            self.example_4_hybrid_search()
            self.example_5_hybrid_search_with_filters()
            self.example_6_keyword_search()
            self.example_7_vector_search()
            self.example_8_complex_filters()
            self.example_9_facet_analysis()
            self.example_10_advanced_queries()
            
            print("\nAll examples completed successfully!")
            
        except Exception as e:
            print(f"\nError running examples: {str(e)}")

def main():
    """Main function to run search examples"""
    try:
        examples = SearchExamples()
        examples.run_all_examples()
        
    except Exception as e:
        print(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()

