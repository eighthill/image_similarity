import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import tkinter as tk
from tkinter import ttk
import logging
import time
from pathlib import Path

import config
from config import LOG_FILE, LOGGER_ACTIVE, DB_PATH, EMB_INDEX
from database.database import Database
from database.repository import ImageRepository
from embedding.embedding_index import EmbeddingIndex

from view.similarity_viewer import SimilarityViewer
from view.view_repository import initialize_database, process_images, backfill_color_histograms, repair_similarity_values, embedding_builder

#DB_PATH = "/Volumes/Extreme SSD/data/images.db/"
#PROJECT_ROOT = Path(__file__).resolve().parent.parent
#EMB_INDEX = PROJECT_ROOT / "embeddings.index"
#LOG_FILE = PROJECT_ROOT / "startup.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

logger = logging.getLogger(__name__)

def create_loading_window():
    loading = tk.Toplevel()
    loading.title("Image Similarity")
    loading.geometry("400x150")
    loading.resizable(False, False)

    loading.update_idletasks()

    width = loading.winfo_width()
    height = loading.winfo_height()

    x = (loading.winfo_screenwidth() - width) // 2
    y = (loading.winfo_screenheight() - height) // 2

    loading.geometry(
        f"{width}x{height}+{x}+{y}"
    )

    ttk.Label(
        loading,
        text="Image Similarity wird geladen...",
        font=("TkDefaultFont", 13, "bold")
    ).pack(pady=(25, 10))

    progress = ttk.Progressbar(
        loading,
        orient="horizontal",
        length=320,
        mode="determinate",
        maximum=100
    )

    progress.pack()

    status = ttk.Label(
        loading,
        text="Starte..."
    )

    status.pack(pady=(8, 0))

    loading.update()

    return loading, progress, status

def main():

    if LOGGER_ACTIVE:
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
        
        root = tk.Tk()
        root.withdraw()

        loading, progress, status = create_loading_window()

        def update_progress(value, text):
            progress["value"] = value
            status.config(text=text)
            loading.update()
            
        update_progress(
            5,
            "Initialisiere Datenbank..."
        )
        
        db = Database(DB_PATH)
        
        initialize_database(db)
        
        update_progress(
            15,
            "Bereite Bilddatenbank vor..."
        )
        
        repo = ImageRepository(db)
        
        update_progress(
            25,
            "Importiere Bilder... 0 / ?"
        )

        process_images(
            repo,
            progress_callback=lambda current, total, text:
                update_progress(
                    15 + (current / total) * 35,
                    f"{text} {current:,} / {total:,}"
                )
        )
        
        update_progress(
            50,
            "Berechne Farbhistogramme..."
        )
        
        backfill_color_histograms(
            repo,
            progress_callback=lambda current, total, text:
                update_progress(
                    50 + (current / total) * 10,
                    f"{text} {current:,} / {total:,}"
                )
        )
        
        update_progress(
            60,
            "Repariere Similarity-Werte..."
        )
        
        repair_similarity_values(db)
        
        update_progress(
            70,
            "Lade Embedding Index..."
        )
        
        embedding_index = embedding_builder(
            repo,
            EMB_INDEX
        )
        
        update_progress(
            100,
            "Starte Oberfläche..."
        )
        
        time.sleep(0.2)

        loading.destroy()

        root.deiconify()

    SimilarityViewer(
        root,
        repo,
        embedding_index
    )

    root.mainloop()

if __name__ == "__main__":
    main()
