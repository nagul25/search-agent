import logging

from logging.handlers import RotatingFileHandler
import os

os.makedirs("logs", exist_ok=True)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler (rotates log file when it exceeds 5MB, keeps 3 backups)
file_handler = RotatingFileHandler("logs/app.log", maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Default level can be DEBUG, INFO, ERROR
logger.addHandler(file_handler)
logger.addHandler(console_handler)