import os
from dotmap import DotMap
from dotenv import load_dotenv
from pathlib import Path

CURRENT_MODULE_DIR = os.path.dirname(__file__)
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_MODULE_DIR, "json", "config.json")
SCHEMA_DEFAULTS_PATH = os.path.join(CURRENT_MODULE_DIR, "json", "schema.json")

ERROR_CODES = DotMap(
    {
        "MULTI_BUBBLE_WARN": 1,
        "NO_MARKER_ERR": 2,
    }
)

QTYPE_DATA = {
    "QTYPE_MED": {"vals": ["E", "H"], "orient": "V"},
    "QTYPE_ROLL": {"vals": range(10), "orient": "V"},
    "QTYPE_INT": {"vals": range(10), "orient": "V"},
    "QTYPE_INT_11": {"vals": range(11), "orient": "V"},
    "QTYPE_MCQ4": {"vals": ["A", "B", "C", "D"], "orient": "H"},
    "QTYPE_MCQ5": {"vals": ["A", "B", "C", "D", "E"], "orient": "H"},
}

TEXT_SIZE = 0.95
CLR_BLACK = (50, 150, 150)
CLR_WHITE = (250, 250, 250)
CLR_GRAY = (130, 130, 130)
CLR_DARK_GRAY = (100, 100, 100)
GLOBAL_PAGE_THRESHOLD_WHITE = 200
GLOBAL_PAGE_THRESHOLD_BLACK = 100
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, "images")
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}

load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.environ.get("SECRET_KEY")
