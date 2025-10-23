"""
Azure Blob Storage Integration for CSV Processing
Handles uploading, downloading, and processing CSV files from Azure Blob Storage
"""

import os
import pandas as pd
from typing import Optional, List, Dict, Any
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureBlobIntegration:
    def __init__(self):
        # Azure Blob Storage configuration
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("BLOB_CONTAINER_NAME", "csv-data")
        
        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is required")
        
        # Initialize blob service client
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        
        logger.info("Azure Blob Storage client initialized")
    
    def create_container_if_not_exists(self, container_name: Optional[str] = None):
        """Create blob container if it doesn't exist"""
        try:
            container_name = container_name or self.container_name
            self.blob_service_client.create_container(container_name)
            logger.info(f"Created container: {container_name}")
        except Exception as e:
            if "ContainerAlreadyExists" in str(e):
                logger.info(f"Container {container_name} already exists")
            else:
                logger.error(f"Error creating container: {str(e)}")
                raise
    
    def upload_csv_to_blob(self, 
                          local_file_path: str, 
                          blob_name: Optional[str] = None,
                          container_name: Optional[str] = None) -> str:
        """
        Upload CSV file to Azure Blob Storage
        
        Args:
            local_file_path: Path to local CSV file
            blob_name: Name for the blob (default: same as filename)
            container_name: Container name (default: from env)
        
        Returns:
            Blob URL
        """
        try:
            container_name = container_name or self.container_name
            blob_name = blob_name or os.path.basename(local_file_path)
            
            # Create container if it doesn't exist
            self.create_container_if_not_exists(container_name)
            
            # Upload file
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            blob_url = blob_client.url
            logger.info(f"Uploaded {local_file_path} to {blob_url}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Error uploading CSV to blob: {str(e)}")
            raise
    
    def download_csv_from_blob(self, 
                              blob_name: str, 
                              local_file_path: Optional[str] = None,
                              container_name: Optional[str] = None) -> str:
        """
        Download CSV file from Azure Blob Storage
        
        Args:
            blob_name: Name of the blob to download
            local_file_path: Local path to save file (default: ./downloaded_{blob_name})
            container_name: Container name (default: from env)
        
        Returns:
            Local file path
        """
        try:
            container_name = container_name or self.container_name
            local_file_path = local_file_path or f"./downloaded_{blob_name}"
            
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            logger.info(f"Downloaded {blob_name} to {local_file_path}")
            return local_file_path
            
        except ResourceNotFoundError:
            logger.error(f"Blob {blob_name} not found in container {container_name}")
            raise
        except Exception as e:
            logger.error(f"Error downloading CSV from blob: {str(e)}")
            raise
    
    def list_blobs(self, 
                   container_name: Optional[str] = None,
                   prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List blobs in container
        
        Args:
            container_name: Container name (default: from env)
            prefix: Filter blobs by prefix
        
        Returns:
            List of blob information
        """
        try:
            container_name = container_name or self.container_name
            container_client = self.blob_service_client.get_container_client(container_name)
            
            blobs = []
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blobs.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified,
                    "content_type": blob.content_settings.content_type if blob.content_settings else None,
                    "url": f"{self.blob_service_client.url}/{container_name}/{blob.name}"
                })
            
            logger.info(f"Found {len(blobs)} blobs in container {container_name}")
            return blobs
            
        except Exception as e:
            logger.error(f"Error listing blobs: {str(e)}")
            raise
    
    def delete_blob(self, 
                   blob_name: str, 
                   container_name: Optional[str] = None) -> bool:
        """
        Delete blob from container
        
        Args:
            blob_name: Name of blob to delete
            container_name: Container name (default: from env)
        
        Returns:
            True if deleted successfully
        """
        try:
            container_name = container_name or self.container_name
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            blob_client.delete_blob()
            logger.info(f"Deleted blob: {blob_name}")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"Blob {blob_name} not found")
            return False
        except Exception as e:
            logger.error(f"Error deleting blob: {str(e)}")
            raise
    
    def get_blob_properties(self, 
                           blob_name: str, 
                           container_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get blob properties
        
        Args:
            blob_name: Name of blob
            container_name: Container name (default: from env)
        
        Returns:
            Blob properties
        """
        try:
            container_name = container_name or self.container_name
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            properties = blob_client.get_blob_properties()
            
            return {
                "name": blob_name,
                "size": properties.size,
                "last_modified": properties.last_modified,
                "content_type": properties.content_settings.content_type if properties.content_settings else None,
                "etag": properties.etag,
                "url": blob_client.url
            }
            
        except ResourceNotFoundError:
            logger.error(f"Blob {blob_name} not found")
            raise
        except Exception as e:
            logger.error(f"Error getting blob properties: {str(e)}")
            raise
    
    def process_csv_from_blob(self, 
                             blob_name: str, 
                             container_name: Optional[str] = None) -> pd.DataFrame:
        """
        Download and process CSV from blob storage
        
        Args:
            blob_name: Name of CSV blob
            container_name: Container name (default: from env)
        
        Returns:
            Pandas DataFrame
        """
        try:
            # Download CSV
            local_path = self.download_csv_from_blob(blob_name, container_name=container_name)
            
            # Read CSV into DataFrame
            df = pd.read_csv(local_path)
            
            # Clean up local file
            os.remove(local_path)
            
            logger.info(f"Processed CSV from blob {blob_name}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error processing CSV from blob: {str(e)}")
            raise
    
    def upload_dataframe_as_csv(self, 
                               df: pd.DataFrame, 
                               blob_name: str,
                               container_name: Optional[str] = None) -> str:
        """
        Upload DataFrame as CSV to blob storage
        
        Args:
            df: Pandas DataFrame
            blob_name: Name for the blob
            container_name: Container name (default: from env)
        
        Returns:
            Blob URL
        """
        try:
            container_name = container_name or self.container_name
            
            # Create container if it doesn't exist
            self.create_container_if_not_exists(container_name)
            
            # Convert DataFrame to CSV string
            csv_string = df.to_csv(index=False)
            
            # Upload to blob
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            blob_client.upload_blob(csv_string.encode('utf-8'), overwrite=True)
            
            blob_url = blob_client.url
            logger.info(f"Uploaded DataFrame as CSV to {blob_url}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Error uploading DataFrame as CSV: {str(e)}")
            raise

def main():
    """Main function to demonstrate blob operations"""
    try:
        # Initialize blob integration
        blob_integration = AzureBlobIntegration()
        
        # Example operations
        print("Azure Blob Storage Integration Examples")
        print("=" * 50)
        
        # List existing blobs
        print("\n1. Listing existing blobs:")
        blobs = blob_integration.list_blobs()
        for blob in blobs[:5]:  # Show first 5
            print(f"  File: {blob['name']} ({blob['size']} bytes)")
        
        # Upload local CSV file
        print("\n2. Uploading local CSV file:")
        local_csv = "technology_standard_list.csv"
        if os.path.exists(local_csv):
            blob_url = blob_integration.upload_csv_to_blob(local_csv)
            print(f"  Uploaded to: {blob_url}")
        else:
            print(f"  Warning: Local file {local_csv} not found")
        
        # Download and process CSV
        print("\n3. Downloading and processing CSV:")
        blob_name = "technology_standard_list.csv"
        try:
            df = blob_integration.process_csv_from_blob(blob_name)
            print(f"  Processed CSV: {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns: {', '.join(df.columns[:5])}...")
        except Exception as e:
            print(f"  Error: {str(e)}")
        
    except Exception as e:
        print(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    main()

