import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SECRET_KEY = os.environ.get("SECRET_KEY")
ANSWER_KEY = {0: 2, 1: 4, 2: 0, 3: 1, 4: 1}
UPLOAD_SERVICE = os.environ.get("UPLOAD_SERVICE")
HOST = os.environ.get("HOST")
PORT = int(os.environ.get("PORT"))
USER = os.environ.get("POSTGRES_USER")
PASSWORD = os.environ.get("POSTGRES_PASSWORD")
CONNECTION_STRING = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/mydb"
SECRET_KEY = os.environ.get("SECRET_KEY")
ALLOWED_EXTENSIONS = ("txt", "jpg", "png", "tiff")
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, "images")
UPLOAD_SERVICE_ENDPOINT = UPLOAD_SERVICE + "/image-api/upload-image"
