import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import tkinter as tk
import os
from pathlib import Path

from database.database import Database
from database.repository import ImageRepository
from embedding_index import EmbeddingIndex

from view.similarity_viewer import SimilarityViewer
from view.view_repository import initialize_database, process_images, backfill_color_histograms, repair_similarity_values, embedding_builder

DB_PATH = "/Volumes/Extreme SSD/data/images.db"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_INDEX = PROJECT_ROOT / "embeddings.index"

def main():
    db = Database(DB_PATH)

    initialize_database(db)

    repo = ImageRepository(db)

    process_images(repo)

    backfill_color_histograms(repo)

    repair_similarity_values(db)

    print("Finished importing images.")

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
