import os
from typing import List, Optional
from azure.storage.blob.aio import BlobServiceClient
from fastapi import UploadFile
from pathlib import Path
from urllib.parse import urlparse, unquote
from config import Config
from app.log_config import logger

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
            logger.info(f"Created container: {container_name}")
        except Exception as create_error:
            logger.info(f"Container {container_name} may already exist: {create_error}")
            pass
        for file in files:
            blob_client = container_client.get_blob_client(blob=file.filename)
            content = await file.read()
            file_contents.append({"filename": file.filename, "content_type": file.content_type})
            logger.info(f"Read file: {file.filename}, content_type: {file.content_type}")
            await blob_client.upload_blob(
                data=content, 
                metadata={"content_type": file.content_type}, 
                overwrite=True
                )
            uploaded_files.append({
                "filename": file.filename,
                "blob_url": blob_client.url
            })
            try:
                await file.close()
            except Exception as close_error:
                logger.error(f"Error closing file {file.filename}: {close_error}")
                pass
            logger.info(f"Uploaded blob: {file.filename} to container: {container_name}, blob url: {blob_client.url}")
        return {"uploaded_files": uploaded_files, "message": "Files uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading blob: {e}")
        raise
    finally:
        await blob_service_client.close()
        logger.info("Closing BlobServiceClient")

async def upload_png_to_blob(png_directory: str, file_name: str):
    blob_service_client = await get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)
    logger.info(f"uploading pngs from directory: {png_directory} - under file: {file_name}")
    uploaded_pngs = []
    try:
        try:
            await container_client.create_container()
            logger.info(f"Created container: {container_name}")
        except Exception as create_error:
            logger.info(f"Container {container_name} may already exist: {create_error}")
            pass

        for index, png_file in enumerate(os.listdir(png_directory)):
            if png_file.endswith(".png"):
                png_path = os.path.join(png_directory, png_file)
                blob_path = f"{file_name}/{index}_{png_file}"
                blob_client = container_client.get_blob_client(blob=blob_path)
                with open(png_path, "rb") as data:
                    await blob_client.upload_blob(data, overwrite=True)
                logger.info(f"Uploaded PNG blob: {blob_path} to container: {container_name}, blob url: {blob_client.url}")
                uploaded_pngs.append({
                    "filename": png_file,
                    "blob_url": blob_client.url
                })
        return uploaded_pngs
    except Exception as e:
        logger.error(f"Error uploading PNGs to blob: {e}")
        raise
    finally:
        try:
            await blob_service_client.close()
        except Exception as close_err:
            logger.warning(f"Error closing BlobServiceClient: {close_err}")
        logger.info("Closing BlobServiceClient")


async def download_blob_to_local(blob_url: str, local_path: str, *, use_url_container: bool = True):
    blob_service_client = await get_blob_service_client()
    logger.info(f"Downloading blob from {blob_url}")
    try:
        # Extract blob name from URL if full URL is passed
        if blob_url.startswith("http"):
            parsed_url = urlparse(blob_url)
            path_parts = [part for part in parsed_url.path.split("/") if part]
            if len(path_parts) < 2:
                raise ValueError(f"Invalid blob URL format: {parsed_url.path}")
            
            url_container = path_parts[0]
            encoded_blob_name = "/".join(path_parts[1:])
            blob_name = unquote(encoded_blob_name)
            resolved_container = url_container if use_url_container else container_name
            logger.info(f"Resolved container: {resolved_container}, blob name: {blob_name}")
        else:
            blob_name = blob_url
            resolved_container = container_name
            logger.info(f"Using default container: {resolved_container}, blob name: {blob_name}")

        # Ensure local directories exist
        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        blob_client = blob_service_client.get_blob_client(container=resolved_container, blob=blob_name)

        downloader = await blob_client.download_blob()
        # Stream to file in chunks
        with local_path_obj.open("wb") as download_file:
            async for chunk in downloader.chunks():
                download_file.write(chunk)

        logger.info(f"Successfully downloaded blob to {local_path}")

        return str(local_path_obj)

    except ResourceNotFoundError as rnfe:
        logger.error(f"Blob not found: {blob_name} in container: {resolved_container} {rnfe}")
        raise
    except Exception as e:
        logger.error(f"Error downloading blob: {e}")
        raise
    finally:
        await blob_service_client.close()
        logger.info("Closing BlobServiceClient")