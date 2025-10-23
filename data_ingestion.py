"""
Azure AI Search Data Ingestion Script
Processes CSV data and creates embeddings for hybrid search
"""

import pandas as pd
import json
import uuid
import requests
from typing import List, Dict, Any
import openai
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AzureSearchDataIngestion:
    def __init__(self):
        # Azure AI Search configuration
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_KEY")
        self.search_index_name = "technology-tools-index"
        
        # Azure OpenAI configuration for embeddings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.openai_api_version = os.getenv("OPENAI_API_VERSION")
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        
        # Azure Blob Storage configuration
        self.blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.blob_container_name = os.getenv("BLOB_CONTAINER_NAME", "csv-data")
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Azure clients"""
        try:
            # Azure AI Search clients
            credential = AzureKeyCredential(self.search_key)
            self.index_client = SearchIndexClient(
                endpoint=self.search_endpoint,
                credential=credential
            )
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.search_index_name,
                credential=credential
            )
            
            # OpenAI client will be initialized in create_embedding method
            
            # Azure Blob Storage client
            if self.blob_connection_string:
                self.blob_client = BlobServiceClient.from_connection_string(
                    self.blob_connection_string
                )
            
            print("All Azure clients initialized successfully")
            
        except Exception as e:
            print(f"Error initializing clients: {str(e)}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using Azure OpenAI"""
        try:
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                api_key=self.openai_api_key,
                api_version=self.openai_api_version,
                azure_endpoint=self.openai_endpoint
            )
            
            response = client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            return []
    
    def process_csv_data(self, csv_file_path: str) -> List[Dict[str, Any]]:
        """Process CSV data and prepare for indexing"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            print(f"Loaded CSV with {len(df)} rows")
            
            documents = []
            
            for index, row in df.iterrows():
                # Create combined text for embedding
                combined_text = f"""
                Tool: {row.get('NameofTools', '')}
                Capability: {row.get('Capabilities', '')}
                Sub-capability: {row.get('SubCapability', '')}
                Manufacturer: {row.get('Manufacturer', '')}
                TEB Status: {row.get('TEBStatus', '')}
                Description: {row.get('Description', '')}
                Meta Tags: {row.get('MetaTags', '')}
                Meta Tags Description: {row.get('MetaTagsDescription', '')}
                Standards Comments: {row.get('StandardsComments', '')}
                EA Notes: {row.get('EANotes', '')}
                Capability Manager: {row.get('CapabilityManager', '')}
                """.strip()
                
                # Create document for Azure AI Search
                doc = {
                    "id": str(uuid.uuid4()),
                    "Capabilities": row.get('Capabilities', ''),
                    "SubCapability": row.get('SubCapability', ''),
                    "TEBStatus": row.get('TEBStatus', ''),
                    "NameofTools": row.get('NameofTools', ''),
                    "Version": row.get('Version', ''),
                    "StandardsComments": row.get('StandardsComments', ''),
                    "EANotes": row.get('EANotes', ''),
                    "Manufacturer": row.get('Manufacturer', ''),
                    "StandardCategory": row.get('StandardCategory', ''),
                    "EAReferenceID": row.get('EAReferenceID', ''),
                    "Description": row.get('Description', ''),
                    "MetaTags": row.get('MetaTags', ''),
                    "MetaTagsDescription": row.get('MetaTagsDescription', ''),
                    "CapabilityManager": row.get('CapabilityManager', ''),
                    "combined_text": combined_text
                }
                
                # Create embedding for the combined text
                print(f"Creating embedding for row {index + 1}/{len(df)}: {doc['NameofTools']}")
                embedding = self.create_embedding(combined_text)
                if embedding:
                    doc["content_vector"] = embedding
                
                documents.append(doc)
            
            print(f"Processed {len(documents)} documents")
            return documents
            
        except Exception as e:
            print(f"Error processing CSV data: {str(e)}")
            raise
    
    def upload_documents_batch(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Upload documents to Azure AI Search in batches"""
        try:
            total_docs = len(documents)
            print(f"Uploading {total_docs} documents in batches of {batch_size}")
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                print(f"Uploading batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
                
                result = self.search_client.upload_documents(batch)
                
                # Check for errors
                failed_docs = [doc for doc in result if not doc.succeeded]
                if failed_docs:
                    print(f"Warning: {len(failed_docs)} documents failed to upload")
                    for doc in failed_docs:
                        print(f"   Error: {doc.error_message}")
                else:
                    print(f"Batch uploaded successfully")
            
            print(f"All documents uploaded to Azure AI Search")
            
        except Exception as e:
            print(f"Error uploading documents: {str(e)}")
            raise
    
    def download_csv_from_blob(self, blob_name: str, local_path: str = None) -> str:
        """Download CSV file from Azure Blob Storage"""
        try:
            if not local_path:
                local_path = f"./downloaded_{blob_name}"
            
            blob_client = self.blob_client.get_blob_client(
                container=self.blob_container_name,
                blob=blob_name
            )
            
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            print(f"Downloaded {blob_name} to {local_path}")
            return local_path
            
        except Exception as e:
            print(f"Error downloading CSV from blob: {str(e)}")
            raise
    
    def delete_index(self):
        """Delete the search index if it exists"""
        try:
            self.index_client.delete_index(self.search_index_name)
            print(f"Deleted index '{self.search_index_name}'")
        except Exception as e:
            print(f"Index '{self.search_index_name}' doesn't exist or couldn't be deleted: {str(e)}")
    
    def create_index(self):
        """Create the search index if it doesn't exist"""
        try:
            # Check if index exists
            try:
                self.index_client.get_index(self.search_index_name)
                print(f"Index '{self.search_index_name}' already exists")
                return
            except:
                pass
            
            # Load index schema
            with open('azure_search_index_schema.json', 'r') as f:
                index_schema_dict = json.load(f)
            
            # Create index using the REST API approach
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'api-key': self.search_key
            }
            
            # Create index using REST API
            url = f"{self.search_endpoint}/indexes?api-version=2023-11-01"
            response = requests.post(url, headers=headers, json=index_schema_dict)
            
            if response.status_code == 201:
                print(f"Created index '{self.search_index_name}'")
            elif response.status_code == 409:
                print(f"Index '{self.search_index_name}' already exists")
            else:
                print(f"Error creating index: {response.status_code} - {response.text}")
                raise Exception(f"Failed to create index: {response.text}")
            
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            raise
    
    def run_full_ingestion(self, csv_file_path: str = None, blob_name: str = None):
        """Run the complete data ingestion process"""
        try:
            print("Starting Azure AI Search data ingestion...")
            
            # Step 1: Delete and recreate index with correct dimensions
            self.delete_index()
            self.create_index()
            
            # Step 2: Get CSV data
            if blob_name:
                print(f"Downloading CSV from blob: {blob_name}")
                csv_file_path = self.download_csv_from_blob(blob_name)
            elif not csv_file_path:
                csv_file_path = "technology_standard_list.csv"
            
            # Step 3: Process CSV data
            documents = self.process_csv_data(csv_file_path)
            
            # Step 4: Upload documents
            self.upload_documents_batch(documents)
            
            print("Data ingestion completed successfully!")
            
        except Exception as e:
            print(f"Error in full ingestion: {str(e)}")
            raise

def main():
    """Main function to run the ingestion"""
    try:
        # Initialize ingestion class
        ingestion = AzureSearchDataIngestion()
        
        # Run ingestion with local CSV file
        ingestion.run_full_ingestion(csv_file_path="technology_standard_list.csv")
        
        # Alternative: Run ingestion with blob storage
        # ingestion.run_full_ingestion(blob_name="technology_standard_list.csv")
        
    except Exception as e:
        print(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()

