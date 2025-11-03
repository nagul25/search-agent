from typing import List, Optional
from azure.storage.blob.aio import BlobServiceClient
from fastapi import UploadFile
from config import Config

connection_string = Config.BLOB_CONNECTION_STRING
container_name = Config.BLOB_CONTAINER_NAME

async def get_blob_service_client() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(connection_string)

async def upload_blob(files: Optional[List[UploadFile]]):
    blob_service_client = await get_blob_service_client()
    try:
        container_client = blob_service_client.get_container_client(container_name)
        file_contents = []
        uploaded_files = []
        # create container if not exists
        try:
            await container_client.create_container()
            print(f"Created container: {container_name}")
        except Exception as create_error:
            print(f"Container {container_name} may already exist: {create_error}")
            pass
        for file in files:
            blob_client = container_client.get_blob_client(blob=file.filename)
            content = await file.read()
            file_contents.append({"filename": file.filename, "content_type": file.content_type})
            print(f"Read file: {file.filename}, content_type: {file.content_type}")
            await blob_client.upload_blob(data=content, metadata={"content_type": file.content_type}, overwrite=True)
            uploaded_files.append({
                "filename": file.filename,
                "blob_url": blob_client.url
            })
            try:
                await file.close()
            except Exception as close_error:
                print(f"Error closing file {file.filename}: {close_error}")
                pass
            print(f"Uploaded blob: {file.filename} to container: {container_name}, blob url: {blob_client.url}")
        return {"uploaded_files": uploaded_files, "message": "Files uploaded successfully"}
    except Exception as e:
        print(f"Error uploading blob: {e}")
        raise