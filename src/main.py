import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import tkinter as tk
import logging
import time
from pathlib import Path

from database.database import Database
from database.repository import ImageRepository
from embedding.embedding_index import EmbeddingIndex

from view.similarity_viewer import SimilarityViewer
from view.view_repository import initialize_database, process_images, backfill_color_histograms, repair_similarity_values, embedding_builder

DB_PATH = "/Volumes/Extreme SSD/data/images.db/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_INDEX = PROJECT_ROOT / "embeddings.index"
LOG_FILE = PROJECT_ROOT / "startup.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

logger = logging.getLogger(__name__)
logger_active = False

def main():

    if logger_active:
        total_start = time.perf_counter()

        db = Database(DB_PATH)

        step_start = time.perf_counter()
        initialize_database(db)
        logger.info(
            f"Database initialization: "
            f"{time.perf_counter() - step_start:.2f}s"
        )

        repo = ImageRepository(db)

        step_start = time.perf_counter()
        process_images(repo)
        logger.info(
            f"Image import: "
            f"{time.perf_counter() - step_start:.2f}s"
        )

        step_start = time.perf_counter()
        backfill_color_histograms(repo)
        logger.info(
            f"Histogram backfill: "
            f"{time.perf_counter() - step_start:.2f}s"
        )

        step_start = time.perf_counter()
        repair_similarity_values(db)
        logger.info(
            f"Similarity repair: "
            f"{time.perf_counter() - step_start:.2f}s"
        )

        logger.info("Finished importing images.")

        step_start = time.perf_counter()
        embedding_index = embedding_builder(
            repo,
            EMB_INDEX
        )
        logger.info(
            f"Embedding index: "
            f"{time.perf_counter() - step_start:.2f}s"
        )

        total_time = time.perf_counter() - total_start

        logger.info(
            f"\n\nTOTAL STARTUP TIME: "
            f"{total_time:.2f}s "
            f"({total_time / 60:.2f} min)\n\n"
        )

    else:
        
        db = Database(DB_PATH)
        
        initialize_database(db)
        
        repo = ImageRepository(db)

        process_images(repo)
        
        backfill_color_histograms(repo)
        
        repair_similarity_values(db)
        
        embedding_index = embedding_builder(
            repo,
            EMB_INDEX
        )
        
    root = tk.Tk()

    SimilarityViewer(
        root,
        repo,
        embedding_index
    )

    root.mainloop()

if __name__ == "__main__":
    main()
