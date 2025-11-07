import logging
import uvicorn

from app.main import app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DEBUG=True
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)