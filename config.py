import os

from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    load_dotenv()

    BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING", None)
    BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME", None)
    BLOB_ACCOUNT_NAME = os.getenv("BLOB_ACCOUNT_NAME", None)
    BLOB_ACCOUNT_KEY = os.getenv("BLOB_ACCOUNT_KEY", None)
    SAS_URL = os.getenv("SAS_URL", None)
    SAS_TOKEN = os.getenv("SAS_TOKEN", None)