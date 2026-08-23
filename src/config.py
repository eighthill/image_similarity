from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================
# Database
# =========================

DB_PATH = "/Volumes/Extreme SSD/data/images.db"

# =========================
# Embedding Index - FAISS
# =========================

EMB_INDEX = PROJECT_ROOT / "embeddings.index"

# =========================
# Candidates
# =========================

CANDIDATE_COUNT = 100
MULTIPLE_CANDIDATE_COUNT = 5

# =========================
# GUI
# =========================

PAGE_SIZE = 15
TOP_RESULTS = 5

THUMB_SIZE = (150, 110)
TOP_RESULT_SIZE = (210, 155)

# =========================
# Logging
# =========================

LOG_FILE = PROJECT_ROOT / "startup.log"

# =========================
# Image Location
# =========================

IMAGE_FOLDER = "/Volumes/Extreme SSD/data/image_data/"
LOGGER_ACTIVE = False

# =========================
# Weighting
# =========================

COLOR_WEIGHT = 0
EMBEDDING_WEIGHT = 1
HASH_WEIGHT = 0