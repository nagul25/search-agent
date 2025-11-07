# Azure AI Search with Hybrid Retrieval and RAG System for Technology Tools

This project demonstrates how to index CSV data into Azure AI Search with vector embeddings for hybrid retrieval search capabilities, including filtering and semantic search. It also includes a RAG (Retrieval-Augmented Generation) question-answering system that allows users to interact with the indexed data through natural language queries.

- chat interface
- rag system
- data ingestion

## Features

- **Vector Search**: Semantic similarity search using Azure OpenAI embeddings
- **Keyword Search**: Traditional text-based search
- **Hybrid Search**: Combines vector, keyword, and semantic search
- **Advanced Filtering**: Filter by TEB status, manufacturer, capabilities, etc.
- **Azure Blob Storage Integration**: Process CSV files from blob storage
- **Faceted Search**: Get counts and distributions of field values
- **RAG Question Answering**: Interactive Q&A system with AI-powered answer generation
- **Intelligent Query Analysis**: Automatic extraction of search filters from natural language queries

## Project Structure

```
├── azure_search_index_schema.json    # Search index schema definition
├── data_ingestion.py                 # CSV processing and indexing script
├── hybrid_search_client.py           # Search client with all query types
├── azure_blob_integration.py         # Blob storage integration
├── search_examples.py                # Comprehensive search examples
├── query_analyzer.py                 # LLM-powered query analysis
├── rag_system.py                     # RAG orchestration logic
├── chat_interface.py                 # Interactive Q&A interface
├── technology_standard_list.csv       # Sample CSV data
├── requirements.txt                   # Python dependencies
├── env_example.txt                   # Environment variables template
└── README.md                         # This file
```

## Setup Instructions

### 1. Prerequisites

- Azure subscription
- Azure AI Search service
- Azure OpenAI service (for embeddings)
- Azure Blob Storage account (optional)
- Python 3.8+

### 2. Environment Configuration

1. Copy `env_example.txt` to `.env`
2. Fill in your Azure service credentials:

```bash
# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-service-admin-key

# Azure OpenAI (for embeddings and query analysis)
OPENAI_API_KEY=your-openai-api-key
OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
OPENAI_API_VERSION=2024-02-15-preview
EMBEDDING_MODEL=text-embedding-3-small

# Azure Blob Storage (optional)
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
BLOB_CONTAINER_NAME=csv-data

# Azure AI Foundry (for RAG system with GPT-5)
AZURE_AI_FOUNDRY_ENDPOINT=https://your-foundry-resource.openai.azure.com/
AZURE_AI_FOUNDRY_KEY=your-foundry-api-key
AZURE_AI_FOUNDRY_DEPLOYMENT=gpt-5
AZURE_AI_FOUNDRY_API_VERSION=2024-02-15-preview
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Data Ingestion

```bash
python data_ingestion.py
```

This will:
- Create the search index
- Process the CSV file
- Generate embeddings for each record
- Upload documents to Azure AI Search

### 5. Run Search Examples

```bash
python search_examples.py
```

### 6. Run RAG Question-Answering System

```bash
python chat_interface.py
```

This will start an interactive Q&A session where you can ask questions about technology tools in natural language. The system will:
- Analyze your question to extract search parameters and filters
- Retrieve relevant documents from Azure AI Search using hybrid search
- Generate comprehensive answers using Azure AI Foundry GPT-5
- Display the answer along with source documents and search metadata

Examples of questions you can ask:
- "What TEB approved authentication tools are available?"
- "Show me Google's pub/sub messaging tools"
- "Which DevOps tools are under review?"
- "What security and compliance tools can I use?"

## Search Capabilities

### 1. Filter by TEB Status
```python
# TEB Approved tools
result = search_client.filter_search("teb_status eq 'TEB Approved'")

# TEB Not Approved tools
result = search_client.filter_search("teb_status eq 'TEB Not Approved'")
```

### 2. Filter by Manufacturer
```python
# Google tools
result = search_client.filter_search("manufacturer eq 'Google'")

# Microsoft tools
result = search_client.filter_search("manufacturer eq 'Microsoft'")
```

### 3. Hybrid Search
```python
# Authentication tools
result = search_client.hybrid_search("authentication tools")

# Pub/Sub messaging tools
result = search_client.hybrid_search("pub sub messaging")
```

### 4. Hybrid Search with Filters
```python
# TEB Approved authentication tools
result = search_client.hybrid_search(
    "authentication tools",
    filters="teb_status eq 'TEB Approved'"
)

# Google pub/sub tools
result = search_client.hybrid_search(
    "pub sub messaging",
    filters="manufacturer eq 'Google'"
)
```

### 5. Complex Filters
```python
# Multiple manufacturers
result = search_client.filter_search(
    "manufacturer eq 'Google' or manufacturer eq 'Microsoft'"
)

# Status and capability combination
result = search_client.filter_search(
    "teb_status eq 'TEB Approved' and capabilities eq 'Identity & Access Mgmt'"
)
```

## Example Queries

### Filter Queries
- **TEB Approved tools**: `teb_status eq 'TEB Approved'`
- **Google tools**: `manufacturer eq 'Google'`
- **DevOps tools**: `capabilities eq 'DevOps'`
- **Authentication tools**: `capabilities eq 'Identity & Access Mgmt'`

### Hybrid Search Queries
- **Authentication tools**: `"authentication tools"`
- **Pub/Sub tools**: `"pub sub messaging"`
- **Monitoring tools**: `"monitoring observability"`
- **Data engineering**: `"data engineering ETL"`

### Combined Queries
- **TEB Approved pub/sub tools**: Hybrid search for "pub sub" + filter `teb_status eq 'TEB Approved'`
- **Google authentication tools**: Hybrid search for "authentication" + filter `manufacturer eq 'Google'`
- **Approved DevOps tools**: Hybrid search for "DevOps CI/CD" + filter `teb_status eq 'TEB Approved'`

## Index Schema

The search index includes the following fields:

- **Searchable Fields**: `capabilities`, `subcapability`, `tool_name`, `manufacturer`, `description`, `meta_tags`, `combined_text`
- **Filterable Fields**: `teb_status`, `manufacturer`, `capabilities`, `subcapability`, `version`, `standard_category`
- **Facetable Fields**: `teb_status`, `manufacturer`, `capabilities`, `subcapability`, `version`, `standard_category`
- **Vector Field**: `content_vector` (1536 dimensions using text-embedding-3-small)

## Azure Blob Storage Integration

The system supports processing CSV files from Azure Blob Storage:

```python
# Upload CSV to blob storage
blob_integration.upload_csv_to_blob("technology_standard_list.csv")

# Process CSV from blob storage
ingestion.run_full_ingestion(blob_name="technology_standard_list.csv")
```

## Performance Considerations

- **Batch Processing**: Documents are uploaded in batches of 100
- **Embedding Generation**: Uses Azure OpenAI text-embedding-3-small model
- **Index Optimization**: Configured for hybrid search with semantic ranking
- **Vector Search**: Uses HNSW algorithm with cosine similarity

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Verify your Azure service credentials
2. **Index Creation Fails**: Check Azure AI Search service limits
3. **Embedding Errors**: Ensure Azure OpenAI service is properly configured
4. **Blob Access Issues**: Verify blob storage connection string and permissions

### Debug Mode

Enable detailed logging by setting the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## RAG Question-Answering System

The RAG system provides an interactive way to query the indexed technology tools using natural language. It consists of three main components:

### 1. Query Analyzer (`query_analyzer.py`)
- Uses LLM to analyze natural language questions
- Extracts search keywords and intent
- Generates OData filter expressions automatically
- Handles complex queries with multiple conditions

### 2. RAG Orchestrator (`rag_system.py`)
- Orchestrates the complete RAG pipeline
- Integrates query analysis, document retrieval, and answer generation
- Formats retrieved documents as context for the LLM
- Generates comprehensive answers using Azure AI Foundry GPT-5

### 3. Chat Interface (`chat_interface.py`)
- Provides terminal-based interactive Q&A session
- Displays formatted responses with:
  - AI-generated answers
  - Source documents with relevance scores
  - Search metadata (applied filters, query terms, etc.)

### How It Works

1. **Question Input**: User asks a natural language question
2. **Query Analysis**: LLM analyzes the question to extract:
   - Search keywords for hybrid search
   - OData filter expressions (e.g., TEB status, manufacturer, capability)
   - Intent classification
3. **Document Retrieval**: Azure AI Search performs hybrid search with extracted parameters
4. **Context Augmentation**: Retrieved documents are formatted as context
5. **Answer Generation**: Azure AI Foundry GPT-5 generates answer based on context
6. **Response Display**: Formatted answer with sources and metadata shown to user

### Usage Example

```python
from rag_system import RAGSystem

rag = RAGSystem()
result = rag.answer_question("What TEB approved authentication tools are available?")

print(result["answer"])
for source in result["sources"]:
    print(f"- {source['NameofTools']} ({source['Manufacturer']})")
```

## Support

For issues and questions:
1. Check the Azure AI Search documentation
2. Verify your Azure service configurations
3. Review the example queries in `search_examples.py`
4. Test the RAG system with: `python chat_interface.py`

## License

This project is provided as-is for demonstration purposes.





### Run the fastapi application

- Keep the updated .env file
- Do pip install
- python run.py to start the api application
- You can access the swagger using http://localhost:8000/docs url
- Call the ```http://localhost:8000/api/poc/query``` endpoint which will trigger the AI code.
- Arguments for this endpoint 
-   => "query" - string, required 
-   => "files" - optional