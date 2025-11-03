"""
Interactive Chat Interface for RAG System
Terminal-based question answering interface
"""

import sys
from app.services.rag_system import RAGSystem

class ChatInterface:
    def __init__(self):
        print("Initializing RAG System...")
        try:
            self.rag_system = RAGSystem()
            print("System initialized successfully!\n")
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            sys.exit(1)
    
    def format_response(self, result: dict):
        """Format the RAG system response for display"""
        
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(result.get("answer", "No answer generated"))
        
        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)
        
        sources = result.get("sources", [])
        if sources:
            for i, source in enumerate(sources, 1):
                print(f"\n{i}. {source.get('NameofTools', 'N/A')}")
                print(f"   Manufacturer: {source.get('Manufacturer', 'N/A')}")
                print(f"   TEB Status: {source.get('TEBStatus', 'N/A')}")
                print(f"   Capability: {source.get('Capabilities', 'N/A')}")
                if source.get('SubCapability'):
                    print(f"   Sub-Capability: {source.get('SubCapability')}")
                if '@search.score' in source:
                    print(f"   Relevance Score: {source.get('@search.score', 'N/A')}")
        else:
            print("No sources retrieved")
        
        print("\n" + "=" * 80)
        print("SEARCH METADATA")
        print("=" * 80)
        metadata = result.get("metadata", {})
        print(f"Intent: {metadata.get('intent', 'N/A')}")
        print(f"Search Query: {metadata.get('search_query', 'N/A')}")
        filters = metadata.get('filters', 'None')
        print(f"Filters Applied: {filters}")
        print(f"Documents Retrieved: {metadata.get('documents_retrieved', 0)}")
        print("=" * 80 + "\n")
    
    def run(self):
        """Main interactive loop"""
        
        print("Welcome to the Technology Tools Q&A System")
        print("Ask questions about technology tools, capabilities, manufacturers, and TEB status.")
        print("Type 'quit', 'exit', or press Ctrl+C to exit.\n")
        
        while True:
            try:
                question = input("Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Technology Tools Q&A System. Goodbye!")
                    break
                
                print("\nProcessing your question...")
                result = self.rag_system.answer_question(question, top_k=5)
                
                if "error" in result.get("answer", "").lower():
                    print(f"\nError: {result['answer']}")
                else:
                    self.format_response(result)
                
            except KeyboardInterrupt:
                print("\n\nThank you for using the Technology Tools Q&A System. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")

def main():
    """Main entry point"""
    interface = ChatInterface()
    interface.run()

if __name__ == "__main__":
    main()

