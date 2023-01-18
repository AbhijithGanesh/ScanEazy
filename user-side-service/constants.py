import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SECRET_KEY = os.environ.get("SECRET_KEY")
UPLOAD_SERVICE = os.environ.get("UPLOAD_SERVICE")
CONNECTION_STRING = os.environ.get("POSTGRES_CONNECTION_STRING")
SECRET_KEY = os.environ.get("SECRET_KEY")
ALLOWED_EXTENSIONS = ("txt", "jpg", "png", "tiff")
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, "images")
UPLOAD_SERVICE_ENDPOINT = UPLOAD_SERVICE + "/image-api/upload-image"
