#!/usr/bin/env python3
"""
Standards Setup Script

This script performs one-time setup for the Assessment Bot:
1. Uploads static standard documents (PDFs) from the files/ folder to Azure
2. Creates a permanent assistant with file_search capability
3. Saves file IDs and assistant ID for use by the application

Run this script once before starting the application:
    python standards_setup.py

Prerequisites:
- Set environment variables in .env file:
  - AZURE_AI_FOUNDRY_ENDPOINT
  - AZURE_AI_FOUNDRY_KEY
  - AZURE_AI_FOUNDRY_DEPLOYMENT (optional, defaults to gpt-5)
  - AZURE_AI_FOUNDRY_API_VERSION (optional)
- Place standard PDF documents in the files/ folder
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.services.azure_assistant import AzureAssistantClient, ASSESSMENT_SYSTEM_PROMPT
from app.log_config import logger

# Configuration
STANDARDS_FOLDER = os.path.join(PROJECT_ROOT, "files")
STANDARD_FILES_JSON = os.path.join(PROJECT_ROOT, "standard_files.json")
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]


def find_standard_files() -> list:
    """
    Find all standard document files in the files/ folder.
    
    Returns:
        List of file paths to standard documents
    """
    files = []
    
    if not os.path.exists(STANDARDS_FOLDER):
        logger.warning(f"Standards folder not found: {STANDARDS_FOLDER}")
        os.makedirs(STANDARDS_FOLDER, exist_ok=True)
        logger.info(f"Created standards folder: {STANDARDS_FOLDER}")
        return files
    
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(STANDARDS_FOLDER, f"*{ext}")
        found = glob.glob(pattern)
        files.extend(found)
        
        # Also check subdirectories
        pattern_recursive = os.path.join(STANDARDS_FOLDER, "**", f"*{ext}")
        found_recursive = glob.glob(pattern_recursive, recursive=True)
        for f in found_recursive:
            if f not in files:
                files.append(f)
    
    logger.info(f"Found {len(files)} standard files in {STANDARDS_FOLDER}")
    for f in files:
        logger.info(f"  - {os.path.basename(f)}")
    
    return files


def load_existing_config() -> dict:
    """
    Load existing configuration from standard_files.json if it exists.
    
    Returns:
        Dictionary with existing configuration or empty dict
    """
    if os.path.exists(STANDARD_FILES_JSON):
        try:
            with open(STANDARD_FILES_JSON, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing config: {e}")
    return {}


def save_config(config: dict) -> None:
    """
    Save configuration to standard_files.json.
    
    Args:
        config: Configuration dictionary to save
    """
    try:
        with open(STANDARD_FILES_JSON, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {STANDARD_FILES_JSON}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise


def setup_standards():
    """
    Main setup function that:
    1. Finds standard files in files/ folder
    2. Uploads them to Azure
    3. Creates the assistant with file_search
    4. Saves configuration
    """
    print("\n" + "=" * 60)
    print("Assessment Bot - Standards Setup")
    print("=" * 60 + "\n")
    
    # Check environment variables
    required_vars = ["AZURE_AI_FOUNDRY_ENDPOINT", "AZURE_AI_FOUNDRY_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set these in your .env file:")
        for var in missing_vars:
            print(f"  {var}=your_value_here")
        sys.exit(1)
    
    # Find standard files
    print("Step 1: Finding standard files...")
    standard_files = find_standard_files()
    
    if not standard_files:
        print(f"\nNo standard files found in {STANDARDS_FOLDER}")
        print("Please add your standard documents (PDF, TXT, MD, DOCX) to this folder.")
        print("\nYou can still create an assistant without standards and add them later.")
        
        response = input("\nCreate assistant without standards? (y/n): ").strip().lower()
        if response != "y":
            print("Setup cancelled.")
            sys.exit(0)
    
    # Initialize Azure client
    print("\nStep 2: Connecting to Azure AI Foundry...")
    try:
        client = AzureAssistantClient()
        print("  ✓ Connected successfully")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        sys.exit(1)
    
    # Upload standard files
    file_ids = []
    file_mapping = {}
    
    if standard_files:
        print("\nStep 3: Uploading standard files to Azure...")
        for file_path in standard_files:
            filename = os.path.basename(file_path)
            try:
                print(f"  Uploading: {filename}...", end=" ")
                file_id = client.upload_file(file_path, purpose="assistants")
                file_ids.append(file_id)
                file_mapping[filename] = file_id
                print(f"✓ ({file_id})")
            except Exception as e:
                print(f"✗ Failed: {e}")
    else:
        print("\nStep 3: Skipped (no standard files)")
    
    # Create assistant
    print("\nStep 4: Creating assessment assistant...")
    try:
        assistant_id = client.create_assistant(
            name="AssessmentAgent",
            instructions=ASSESSMENT_SYSTEM_PROMPT,
            file_ids=file_ids if file_ids else None
        )
        print(f"  ✓ Assistant created: {assistant_id}")
    except Exception as e:
        print(f"  ✗ Failed to create assistant: {e}")
        
        # Clean up uploaded files on failure
        if file_ids:
            print("\nCleaning up uploaded files...")
            client.delete_files_batch(file_ids)
        sys.exit(1)
    
    # Save configuration
    print("\nStep 5: Saving configuration...")
    config = {
        "assistant_id": assistant_id,
        "files": file_ids,
        "file_mapping": file_mapping,
        "model": client.model,
        "created_at": datetime.utcnow().isoformat()
    }
    save_config(config)
    print(f"  ✓ Saved to {STANDARD_FILES_JSON}")
    
    # Print summary and next steps
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nAssistant ID: {assistant_id}")
    print(f"Standards uploaded: {len(file_ids)}")
    print(f"\nNext steps:")
    print(f"1. Add this to your .env file:")
    print(f"   AZURE_ASSISTANT_ID={assistant_id}")
    print(f"\n2. Start the application:")
    print(f"   python run.py")
    print("\n" + "=" * 60 + "\n")
    
    return config


def update_standards():
    """
    Update standards for an existing assistant by uploading new files.
    """
    print("\n" + "=" * 60)
    print("Assessment Bot - Update Standards")
    print("=" * 60 + "\n")
    
    existing_config = load_existing_config()
    
    if not existing_config.get("assistant_id"):
        print("No existing assistant found. Running full setup instead.")
        return setup_standards()
    
    print(f"Existing assistant: {existing_config['assistant_id']}")
    print(f"Existing files: {len(existing_config.get('files', []))}")
    
    # Find new files
    standard_files = find_standard_files()
    existing_files = set(existing_config.get("file_mapping", {}).keys())
    new_files = [f for f in standard_files if os.path.basename(f) not in existing_files]
    
    if not new_files:
        print("\nNo new files to upload.")
        return existing_config
    
    print(f"\nNew files to upload: {len(new_files)}")
    for f in new_files:
        print(f"  - {os.path.basename(f)}")
    
    response = input("\nUpload new files? (y/n): ").strip().lower()
    if response != "y":
        print("Update cancelled.")
        return existing_config
    
    # Upload new files
    client = AzureAssistantClient()
    new_file_ids = []
    new_file_mapping = dict(existing_config.get("file_mapping", {}))
    
    for file_path in new_files:
        filename = os.path.basename(file_path)
        try:
            print(f"  Uploading: {filename}...", end=" ")
            file_id = client.upload_file(file_path, purpose="assistants")
            new_file_ids.append(file_id)
            new_file_mapping[filename] = file_id
            print(f"✓ ({file_id})")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    # Update configuration
    all_file_ids = existing_config.get("files", []) + new_file_ids
    config = {
        "assistant_id": existing_config["assistant_id"],
        "files": all_file_ids,
        "file_mapping": new_file_mapping,
        "model": existing_config.get("model", "gpt-5"),
        "created_at": existing_config.get("created_at", "unknown"),
        "updated_at": datetime.utcnow().isoformat()
    }
    save_config(config)
    
    print(f"\n✓ Updated configuration with {len(new_file_ids)} new files")
    print(f"  Total files: {len(all_file_ids)}")
    
    return config


def list_standards():
    """
    List current standards configuration.
    """
    print("\n" + "=" * 60)
    print("Assessment Bot - Current Standards")
    print("=" * 60 + "\n")
    
    config = load_existing_config()
    
    if not config:
        print("No configuration found. Run setup first:")
        print("  python standards_setup.py")
        return
    
    print(f"Assistant ID: {config.get('assistant_id', 'Not set')}")
    print(f"Model: {config.get('model', 'Unknown')}")
    print(f"Created: {config.get('created_at', 'Unknown')}")
    
    if config.get("updated_at"):
        print(f"Updated: {config['updated_at']}")
    
    file_mapping = config.get("file_mapping", {})
    print(f"\nStandard files ({len(file_mapping)}):")
    
    for filename, file_id in file_mapping.items():
        print(f"  - {filename}: {file_id}")
    
    print()


def main():
    """
    Main entry point with command-line argument handling.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup and manage assessment bot standards"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="setup",
        choices=["setup", "update", "list"],
        help="Command to run: setup (default), update, or list"
    )
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_standards()
    elif args.command == "update":
        update_standards()
    elif args.command == "list":
        list_standards()


if __name__ == "__main__":
    main()

