import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import imagehash
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps
from tqdm import tqdm
import time
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from database.database import Database
from database.repository import ImageRepository
from image_loader.generator import image_generator
from embedding.extractor import extract_embeddings
from embedding_index import EmbeddingIndex

from similarity.color_similarity import (color_similarity, calculate_color_histogram, color_similarity_from_histograms)
from similarity.embedding_similarity import embedding_similarity
from similarity.hash_similarity import hash_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
#DB_PATH = str(PROJECT_ROOT / "images.db")
DB_PATH = "/Volumes/Extreme SSD/data/images.db"
IMAGE_FOLDER = "/Volumes/Extreme SSD/data/image_data"
THUMB_SIZE = (150, 110)
TOP_RESULT_SIZE = (210, 155)


def load_display_image(filepath):
    with Image.open(filepath) as source:
        image = ImageOps.exif_transpose(source)
        return image.convert("RGB")


def load_image_for_processing(filepath):
    """Load one source image for the processing pipeline.

    This function is intentionally small and thread-safe so it can be used
    by the I/O ThreadPoolExecutor in process_images().
    Returns (image, error) instead of raising, allowing one bad file to be
    skipped without aborting the entire import.
    """
    try:
        with Image.open(filepath) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
        return image, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def calculate_cpu_features(image):
    """Calculate CPU-side image features for one image.

    The two operations are independent and are grouped into one worker task
    so the caller can run many images concurrently.
    """
    hash_value = imagehash.phash(image)
    color_histogram = calculate_color_histogram(image)
    return str(hash_value), color_histogram


def initialize_database(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            width INTEGER,
            height INTEGER,
            photographer TEXT,
            embedding BLOB,
            hash_value TEXT,
            color_histogram BLOB
        )
    """)
    
    # Bestehende Datenbanken migrieren:
    # color_histogram hinzufügen, falls die Spalte noch nicht existiert.
    columns = db.execute(
        "PRAGMA table_info(images)",
        commit=False
    ).fetchall()

    column_names = {row["name"] for row in columns}

    if "color_histogram" not in column_names:
        db.execute(
            "ALTER TABLE images ADD COLUMN color_histogram BLOB"
        )
        print("Added color_histogram column to images table.")

    db.execute("""
        CREATE TABLE IF NOT EXISTS color_similarities (
            id1 INTEGER,
            id2 INTEGER,
            similarity REAL,
            PRIMARY KEY (id1, id2)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS embedding_similarities (
            id1 INTEGER,
            id2 INTEGER,
            similarity REAL,
            PRIMARY KEY (id1, id2)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS hash_similarities (
            id1 INTEGER,
            id2 INTEGER,
            similarity REAL,
            PRIMARY KEY (id1, id2)
        )
    """)


def _similarity_value_to_float(value):
    if value is None:
        return None

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) == 4:
            return float(np.frombuffer(raw, dtype="<f4")[0])
        if len(raw) == 8:
            return float(np.frombuffer(raw, dtype="<f8")[0])
        raise ValueError(f"Unsupported similarity BLOB length: {len(raw)}")

    return float(value)

def repair_similarity_values(db):
    repaired = 0
    for table in (
        "color_similarities",
        "embedding_similarities",
        "hash_similarities",
    ):
        cursor = db.execute(
            f"SELECT rowid, similarity FROM {table} WHERE typeof(similarity) = 'blob'",
            commit=False,
        )
        rows = cursor.fetchall()
        for row in rows:
            value = _similarity_value_to_float(row["similarity"])
            db.execute(
                f"UPDATE {table} SET similarity = ? WHERE rowid = ?",
                (value, row["rowid"]),
                commit=False,
            )
            repaired += 1

    if repaired:
        db.commit()
        print(f"Repaired {repaired} legacy similarity values in the database.")


def process_images(repo):
    imported = 0
    skipped = 0
    repaired = 0
    processed = 0

    batch_size = 32
    io_workers = min(8, max(2, (os.cpu_count() or 4)))

    existing_images = {
        row["filepath"]: row
        for row in repo.get_existing_image_paths()
    }

    print(f"Loaded {len(existing_images)} existing image records from database.")
    print(f"Image loading workers: {io_workers}")

    # Read files concurrently in small batches. The expensive embedding step
    # remains batch-based so MPS/CUDA can process all images together.
    pending_paths = []

    def flush_paths(paths):
        nonlocal imported, repaired, skipped, processed

        if not paths:
            return

        batch = []
        with ThreadPoolExecutor(max_workers=io_workers) as executor:
            loaded = executor.map(load_image_for_processing, paths)

            for filepath, (image, error) in zip(paths, loaded):
                processed += 1

                if processed % 1000 == 0:
                    print(f"Processed: {processed}")

                if error is not None:
                    print(f"Skipping image: {filepath} -> {error}")
                    continue

                existing = existing_images.get(str(filepath))

                if (
                    existing is not None
                    and existing["has_embedding"]
                    and existing["has_hash"]
                    and existing["has_color_histogram"]
                ):
                    skipped += 1
                    continue

                batch.append((image, str(filepath), existing))

        if batch:
            imported_delta, repaired_delta = process_batch(repo, batch)
            imported += imported_delta
            repaired += repaired_delta
            repo.db.commit()

    for filepath in image_generator(IMAGE_FOLDER):
        filepath = str(filepath)

        existing = existing_images.get(filepath)
        if (
            existing is not None
            and existing["has_embedding"]
            and existing["has_hash"]
            and existing["has_color_histogram"]
        ):
            skipped += 1
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed: {processed}")
            continue

        pending_paths.append(filepath)
        if len(pending_paths) >= batch_size:
            flush_paths(pending_paths)
            pending_paths = []

    flush_paths(pending_paths)
    repo.db.commit()

    print(
        f"Image import: {imported} new, {repaired} repaired, "
        f"{skipped} already complete."
    )


def process_batch(repo, batch):
    imported = 0
    repaired = 0

    print(f"Processing batch of {len(batch)} images")

    images = [item[0] for item in batch]

    # CPU feature extraction is independent of the embedding calculation.
    # Start it in worker threads while the batch embedding runs on MPS/CUDA.
    cpu_start = time.perf_counter()
    feature_workers = min(8, max(2, (os.cpu_count() or 4)))

    with ThreadPoolExecutor(max_workers=feature_workers) as executor:
        feature_futures = [
            executor.submit(calculate_cpu_features, image)
            for image in images
        ]

        embedding_start = time.perf_counter()
        embeddings = extract_embeddings(images)
        embedding_time = time.perf_counter() - embedding_start

        feature_results = [future.result() for future in feature_futures]

    cpu_time = time.perf_counter() - cpu_start
    hashes = [result[0] for result in feature_results]
    color_histograms = [result[1] for result in feature_results]

    db_start = time.perf_counter()

    for ((image, filepath, existing), hash_value, embedding, color_histogram) in zip(
        batch, hashes, embeddings, color_histograms
    ):
        filename = Path(filepath).name
        width, height = image.size

        if existing is None:
            repo.insert_image(
                filename=filename,
                filepath=filepath,
                width=width,
                height=height,
                embedding=embedding.tobytes(),
                hash_value=hash_value,
                color_histogram=color_histogram.tobytes(),
            )
            imported += 1
        else:
            repo.update_image_features(
                existing["id"],
                embedding.tobytes(),
                hash_value,
                color_histogram.tobytes(),
            )
            repaired += 1

    db_time = time.perf_counter() - db_start

    print(
        f"Batch {len(batch)}: "
        f"Embedding {embedding_time:.2f}s | "
        f"CPU features {cpu_time:.2f}s | "
        f"DB {db_time:.2f}s"
    )

    return imported, repaired


def _ordered_pair(id1, id2):
    return (id1, id2) if id1 < id2 else (id2, id1)


def calculate_similarities(repo):
    images = sorted(repo.get_all_images(), key=lambda row: row["id"])

    existing_color = repo.get_existing_similarity_pairs("color_similarities")
    existing_embedding = repo.get_existing_similarity_pairs("embedding_similarities")
    existing_hash = repo.get_existing_similarity_pairs("hash_similarities")

    total_pairs = len(images) * (len(images) - 1) // 2
    complete_pairs = (
        existing_color
        & existing_embedding
        & existing_hash
    )
    missing_pairs = total_pairs - len(complete_pairs)

    print(
        f"Similarity cache: {len(complete_pairs)}/{total_pairs} pairs complete; "
        f"{missing_pairs} pairs need work."
    )

    if missing_pairs == 0:
        print("All similarities are already cached. Nothing to recalculate.")
        return

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            img1 = images[i]
            img2 = images[j]
            pair = (img1["id"], img2["id"])

            color_missing = pair not in existing_color
            embedding_missing = pair not in existing_embedding
            hash_missing = pair not in existing_hash

            if not (color_missing or embedding_missing or hash_missing):
                continue

            print(f"\n{img1['filename']} <-> {img2['filename']}")

            image1 = None
            image2 = None

            if color_missing:
                image1 = cv2.imread(img1["filepath"])
                image2 = cv2.imread(img2["filepath"])
                if image1 is None or image2 is None:
                    print("Skipping color similarity: image could not be read")
                    color_missing = False
                else:
                    color_sim = color_similarity(image1, image2)
                    repo.insert_color_similarity(img1["id"], img2["id"], color_sim)
                    existing_color.add(pair)
                    print(f"Color: {color_sim:.4f}")

            if embedding_missing:
                e1 = np.frombuffer(img1["embedding"], dtype=np.float32)
                e2 = np.frombuffer(img2["embedding"], dtype=np.float32)
                embedding_sim = embedding_similarity(e1, e2)
                repo.insert_embedding_similarity(img1["id"], img2["id"], embedding_sim)
                existing_embedding.add(pair)
                print(f"Embedding: {embedding_sim:.4f}")

            if hash_missing:
                h1 = imagehash.hex_to_hash(img1["hash_value"])
                h2 = imagehash.hex_to_hash(img2["hash_value"])
                hash_sim = hash_similarity(h1, h2)
                repo.insert_hash_similarity(img1["id"], img2["id"], hash_sim)
                existing_hash.add(pair)
                print(f"Hash: {hash_sim:.4f}")

    repo.db.commit()


def load_similarity_map(repo, reference_id):
    rows = {}
    for table, key in [
        ("color_similarities", "color"),
        ("embedding_similarities", "embedding"),
        ("hash_similarities", "hash"),
    ]:
        cursor = repo.db.execute(
            f"SELECT id1, id2, similarity FROM {table} WHERE id1 = ? OR id2 = ?",
            (reference_id, reference_id),
        )
        for row in cursor.fetchall():
            other_id = row["id2"] if row["id1"] == reference_id else row["id1"]
            rows.setdefault(other_id, {})[key] = _similarity_value_to_float(row["similarity"])
    return rows

def calculate_similarities_for_reference(
    repo,
    reference_id,
    embedding_index,
    candidate_count=100,
):

    reference = repo.get_image_by_id(
        reference_id
    )

    if reference is None:
        return {}

    if reference["embedding"] is None:
        return {}

    if reference["color_histogram"] is None:
        return {}

    # Referenz-Embedding
    ref_embedding = np.frombuffer(
    reference["embedding"],
    dtype=np.float32,
    )

    print(
        "index:",
        embedding_index.index.ntotal,
        embedding_index.index.d,
        flush=True,
    )

    print(
        "embedding:",
        ref_embedding.shape,
        ref_embedding.dtype,
        ref_embedding.flags["C_CONTIGUOUS"],
        flush=True,
    )

    print("Before FAISS search", flush=True)

    candidates = embedding_index.search(
        ref_embedding,
        k=candidate_count,
    )

    print(
        "After FAISS search:",
        len(candidates),
        flush=True,
    )

    # Referenz-Hash
    ref_hash = imagehash.hex_to_hash(
        reference["hash_value"]
    )

    # Referenz-Histogramm
    ref_color_histogram = np.frombuffer(
        reference["color_histogram"],
        dtype=np.float32,
    )

    results = {}

    for image_id, embedding_sim in candidates:

        if image_id == reference_id:
            continue

        image = repo.get_image_by_id(
            image_id
        )

        if image is None:
            continue

        if image["color_histogram"] is None:
            continue

        # Hash
        image_hash = imagehash.hex_to_hash(
            image["hash_value"]
        )

        hash_sim = hash_similarity(
            ref_hash,
            image_hash,
        )

        # Color
        image_color_histogram = np.frombuffer(
            image["color_histogram"],
            dtype=np.float32,
        )

        color_sim = color_similarity_from_histograms(
            ref_color_histogram,
            image_color_histogram,
        )

        results[image_id] = {
            "color": color_sim,
            "embedding": embedding_sim,
            "hash": hash_sim,
        }

    return results

class SimilarityViewer:
    def __init__(self, root, repo, images, embedding_index):
        self.root = root
        self.repo = repo
        self.images = images
        self.embedding_index = embedding_index
        
        self.image_by_id = {row["id"]: row for row in images}
        self.photo_refs = {}
        self.selected_id = None
        self.current_results = []
        self.sort_column = "overall"
        self.sort_descending = True

        root.title("Image Similarity Explorer")
        root.geometry("1250x800")
        root.minsize(900, 650)

        outer = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer, padding=10)
        right = ttk.Frame(outer, padding=10)
        outer.add(left, weight=1)
        outer.add(right, weight=2)

        ttk.Label(left, text="Referenzbild auswählen", font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        ttk.Label(left, text="Klick auf ein Bild → rechts werden die ähnlichsten Bilder angezeigt.").pack(anchor="w", pady=(2, 10))

        self.thumb_canvas = tk.Canvas(
            left,
            highlightthickness=0
        )

        scrollbar = ttk.Scrollbar(
            left,
            orient="vertical",
            command=self.thumb_canvas.yview
        )

        self.thumb_frame = ttk.Frame(
            self.thumb_canvas
        )

        self.thumb_window = self.thumb_canvas.create_window(
            (0, 0),
            window=self.thumb_frame,
            anchor="nw"
        )
        
        self.thumb_frame.bind(
            "<Configure>",
            self._on_thumbnail_frame_configure
        )

        self.thumb_canvas.configure(
            yscrollcommand=scrollbar.set
        )

        self.thumb_canvas.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y
        )
        
        self.reference_label = ttk.Label(right, text="Noch kein Referenzbild ausgewählt", font=("TkDefaultFont", 14, "bold"))
        self.reference_label.pack(anchor="w")
        self.reference_score = ttk.Label(right, text="")
        self.reference_score.pack(anchor="w", pady=(2, 8))

        ttk.Label(right, text="Top 5 ähnliche Bilder", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 6))

        self.top5_canvas = tk.Canvas(right, highlightthickness=0, height=340)
        self.top5_scrollbar = ttk.Scrollbar(right, orient="horizontal", command=self.top5_canvas.xview)
        self.top5_frame = ttk.Frame(self.top5_canvas)
        self.top5_window = self.top5_canvas.create_window((0, 0), window=self.top5_frame, anchor="nw")
        self.top5_canvas.configure(xscrollcommand=self.top5_scrollbar.set)
        self.top5_canvas.pack(fill=tk.X, expand=False)
        self.top5_scrollbar.pack(fill=tk.X, pady=(0, 10))
        self.top5_frame.bind("<Configure>", lambda e: self.top5_canvas.configure(scrollregion=self.top5_canvas.bbox("all")))
        self.top5_refs = []

        ttk.Label(right, text="Alle Treffer", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 6))
        results_container = ttk.Frame(right)
        results_container.pack(fill=tk.BOTH, expand=True)

        self.results = ttk.Treeview(
            results_container,
            columns=("image", "overall", "color", "embedding", "hash"),
            show="headings",
            height=8,
        )
        headings = {
            "image": "Bild",
            "overall": "Gesamt",
            "color": "Farbe",
            "embedding": "Embedding",
            "hash": "Hash",
        }
        widths = {"image": 360, "overall": 90, "color": 90, "embedding": 100, "hash": 90}
        for col in headings:
            self.results.heading(
                col,
                text=headings[col],
                command=lambda c=col: self.sort_results(c),
            )
            self.results.column(col, width=widths[col], anchor="center" if col != "image" else "w")
        results_scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=self.results.yview)
        self.results.configure(yscrollcommand=results_scrollbar.set)
        self.results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_lazy_thumbnails()

    def _build_lazy_thumbnails(self):

        self.thumb_columns = 3
        self.thumb_visible_rows = 8

        self.thumb_widgets = {}

        for column in range(self.thumb_columns):
            self.thumb_frame.columnconfigure(
                column,
                weight=1
            )

        self.thumb_canvas.bind(
            "<Configure>",
            self._on_thumbnail_canvas_configure
        )

        self.thumb_canvas.bind(
            "<MouseWheel>",
            self._on_thumbnail_scroll
        )

        self.thumb_frame.bind(
            "<Configure>",
            self._on_thumbnail_frame_configure
        )

        self._update_thumbnail_scrollregion()
        self._update_visible_thumbnails()

    def _update_thumbnail_scrollregion(self):
        """Keep the canvas window at the full virtual thumbnail-list size.

        A canvas window containing a ttk.Frame can otherwise keep the frame
        at its requested size (1x1) while lazy-loading creates only a subset
        of the grid children. That makes the embedded frame/window collapse
        and the thumbnails never get a drawable area.
        """
        total_rows = (
            len(self.images) + self.thumb_columns - 1
        ) // self.thumb_columns

        row_height = 190
        total_height = max(1, total_rows * row_height)
        canvas_width = max(1, self.thumb_canvas.winfo_width())

        # Explicitly size the canvas window AND the embedded frame.
        # This is the important part for lazy-loaded thumbnails.
        self.thumb_canvas.itemconfigure(
            self.thumb_window,
            width=canvas_width,
            height=total_height,
        )

        self.thumb_frame.configure(
            width=canvas_width,
            height=total_height,
        )

        self.thumb_canvas.configure(
            scrollregion=(
                0,
                0,
                canvas_width,
                total_height,
            )
        )
        
    def _update_visible_thumbnails(self):

        if not self.images:
            return

        canvas_height = self.thumb_canvas.winfo_height()

        if canvas_height <= 1:
            self.thumb_canvas.after(
                50,
                self._update_visible_thumbnails
            )
            return

        scroll_top = self.thumb_canvas.canvasy(0)

        row_height = 190

        first_row = max(
            0,
            int(scroll_top // row_height) - 2
        )

        visible_rows = (
            int(canvas_height // row_height) + 4
        )

        last_row = first_row + visible_rows

        first_index = first_row * self.thumb_columns

        last_index = min(
            len(self.images),
            last_row * self.thumb_columns
        )

        visible_indices = set(
            range(first_index, last_index)
        )

        # Nicht mehr benötigte Widgets entfernen
        for index in list(self.thumb_widgets):

            if index not in visible_indices:

                widget = self.thumb_widgets.pop(index)
                widget.destroy()

                self.photo_refs.pop(index, None)

        # Neue Widgets erzeugen
        for index in visible_indices:

            if index not in self.thumb_widgets:

                self._create_thumbnail_widget(
                    index
                )

    def _create_thumbnail_widget(self, index):

        row = self.images[index]

        row_number = index // self.thumb_columns
        column = index % self.thumb_columns

        frame = ttk.Frame(
            self.thumb_frame,
            padding=5,
            relief="ridge"
        )

        frame.grid(
            row=row_number,
            column=column,
            sticky="nsew",
            padx=3,
            pady=3
        )

        try:

            img = load_display_image(
                row["filepath"]
            )

            img.thumbnail(
                THUMB_SIZE
            )

            thumb = Image.new(
                "RGB",
                THUMB_SIZE,
                "white"
            )

            x = (
                THUMB_SIZE[0] - img.width
            ) // 2

            y = (
                THUMB_SIZE[1] - img.height
            ) // 2

            thumb.paste(
                img,
                (x, y)
            )

            photo = ImageTk.PhotoImage(
                thumb
            )

            self.photo_refs[index] = photo

            image_label = tk.Label(
                frame,
                image=photo,
                cursor="hand2"
            )

            image_label.pack()

            image_label.bind(
                "<Button-1>",
                lambda e, image_id=row["id"]:
                    self.select_reference(image_id)
            )

        except Exception as e:
            print(
                f"Thumbnail error [{index}] "
                f"{row['filepath']}: {e}"
            )

            image_label = tk.Label(
                frame,
                text="Bild nicht verfügbar"
            )
            image_label.pack()

        name = ttk.Label(
            frame,
            text=row["filename"],
            wraplength=150,
            justify="center",
            cursor="hand2"
        )

        name.pack(
            fill=tk.X,
            pady=(4, 0)
        )

        name.bind(
            "<Button-1>",
            lambda e, image_id=row["id"]:
                self.select_reference(image_id)
        )

        self.thumb_widgets[index] = frame
        
        if index == 0:
            self.thumb_canvas.after(
                100,
                lambda: print(
                    "DEBUG:",
                    "canvas=",
                    self.thumb_canvas.winfo_width(),
                    self.thumb_canvas.winfo_height(),
                    "frame=",
                    frame.winfo_width(),
                    frame.winfo_height(),
                    "label=",
                    image_label.winfo_width(),
                    image_label.winfo_height(),
                    "window bbox=",
                    self.thumb_canvas.bbox(self.thumb_window),
                )
            )
        
    def _on_thumbnail_scroll(self, event):

        self.thumb_canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )

        self._update_visible_thumbnails()

        return "break"
    
    def _on_thumbnail_canvas_configure(self, event):
        # The canvas width changes with the pane. Recalculate the virtual
        # list dimensions and then update which thumbnails are materialized.
        self._update_thumbnail_scrollregion()
        self._update_visible_thumbnails()
        
    def _on_thumbnail_frame_configure(self, event):
        # Do not derive the scrollregion from the currently materialized
        # children: lazy loading intentionally creates only visible rows.
        # The virtual list size is controlled by _update_thumbnail_scrollregion.
        self._update_thumbnail_scrollregion()

    def _clear_top5(self):
        for widget in self.top5_frame.winfo_children():
            widget.destroy()
        self.top5_refs.clear()

    def _show_top5(self, results, sort_column="overall"):
        self._clear_top5()

        # -------------------------------------------------
        # Referenzbild ganz links anzeigen
        # -------------------------------------------------
        if self.selected_id is not None:
            reference = self.image_by_id[self.selected_id]

            ref_card = ttk.Frame(
                self.top5_frame,
                padding=6,
                relief="ridge"
            )
            ref_card.grid(
                row=1,
                column=0,
                padx=(5, 15),
                pady=3,
                sticky="n"
            )

            try:
                img = load_display_image(reference["filepath"])
                img.thumbnail(TOP_RESULT_SIZE)

                ref_image = Image.new(
                    "RGB",
                    TOP_RESULT_SIZE,
                    "white"
                )

                x = (TOP_RESULT_SIZE[0] - img.width) // 2
                y = (TOP_RESULT_SIZE[1] - img.height) // 2

                ref_image.paste(img, (x, y))

                photo = ImageTk.PhotoImage(ref_image)
                self.top5_refs.append(photo)

                image_label = tk.Label(
                    ref_card,
                    image=photo
                )
                image_label.pack()

            except Exception as exc:
                ttk.Label(
                    ref_card,
                    text=f"Bild nicht verfügbar\n{exc}",
                    width=28,
                    anchor="center"
                ).pack(pady=40)

            ttk.Label(
                ref_card,
                text="REFERENZ",
                font=("TkDefaultFont", 11, "bold")
            ).pack(pady=(5, 2))

            ttk.Label(
                ref_card,
                text=reference["filename"],
                wraplength=210,
                justify="center"
            ).pack(fill=tk.X)

        # -------------------------------------------------
        # Top-5 wie bisher
        # -------------------------------------------------
        index = {
            "overall": 0,
            "image": 1,
            "color": 2,
            "embedding": 3,
            "hash": 4,
        }

        if sort_column == "image":
            ranked_results = sorted(
                results,
                key=lambda row:
                self.image_by_id[row[1]]["filename"].lower()
            )
        else:
            ranked_results = sorted(
                results,
                key=lambda row: row[index[sort_column]],
                reverse=True
            )

        method_names = {
            "overall": "Gesamt",
            "color": "Farbe",
            "embedding": "Embedding",
            "hash": "Hash",
            "image": "Bild",
        }

        ttk.Label(
            self.top5_frame,
            text=f"Top 5 nach {method_names[sort_column]}",
            font=("TkDefaultFont", 11, "bold")
        ).grid(
            row=0,
            column=1,
            columnspan=5,
            sticky="w",
            pady=(0, 5)
        )

        # Die fünf Treffer beginnen jetzt bei column=1
        for rank, (
            overall,
            image_id,
            color,
            embedding,
            hash_value
        ) in enumerate(ranked_results[:5], start=1):

            row = self.image_by_id[image_id]

            card = ttk.Frame(
                self.top5_frame,
                padding=6,
                relief="ridge"
            )
            card.grid(
                row=1,
                column=rank,
                padx=5,
                pady=3,
                sticky="n"
            )

            try:
                img = load_display_image(row["filepath"])
                img.thumbnail(TOP_RESULT_SIZE)

                card_image = Image.new(
                    "RGB",
                    TOP_RESULT_SIZE,
                    "white"
                )

                x = (TOP_RESULT_SIZE[0] - img.width) // 2
                y = (TOP_RESULT_SIZE[1] - img.height) // 2

                card_image.paste(img, (x, y))

                photo = ImageTk.PhotoImage(card_image)
                self.top5_refs.append(photo)

                image_label = tk.Label(
                    card,
                    image=photo,
                    cursor="hand2"
                )
                image_label.pack()

                image_label.bind(
                    "<Button-1>",
                    lambda e, image_id=image_id:
                    self.select_reference(image_id)
                )

            except Exception as exc:
                ttk.Label(
                    card,
                    text=f"Bild nicht verfügbar\n{exc}",
                    width=28,
                    anchor="center"
                ).pack(pady=40)

            ttk.Label(
                card,
                text=f"#{rank}  {row['filename']}",
                wraplength=210,
                justify="center"
            ).pack(fill=tk.X, pady=(5, 2))

            if sort_column == "overall":
                selected_score = overall
            elif sort_column == "color":
                selected_score = color
            elif sort_column == "embedding":
                selected_score = embedding
            elif sort_column == "hash":
                selected_score = hash_value
            else:
                selected_score = None

            if selected_score is not None:
                ttk.Label(
                    card,
                    text=f"{method_names[sort_column]}: {selected_score:.4f}",
                    font=("TkDefaultFont", 10, "bold")
                ).pack()

            ttk.Label(
                card,
                text=(
                    f"Gesamt {overall:.4f}\n"
                    f"Farbe {color:.4f}\n"
                    f"Embedding {embedding:.4f}\n"
                    f"Hash {hash_value:.4f}"
                ),
                justify="center"
            ).pack()

    def _update_heading_labels(self):
        labels = {
            "image": "Bild",
            "overall": "Gesamt",
            "color": "Farbe",
            "embedding": "Embedding",
            "hash": "Hash",
        }
        for col, label in labels.items():
            marker = ""
            if col == self.sort_column:
                marker = "  ▼" if self.sort_descending else "  ▲"
            self.results.heading(col, text=label + marker)

    def sort_results(self, column):
        if not self.current_results:
            return

        if column == self.sort_column:
            self.sort_descending = not self.sort_descending
        else:
            self.sort_column = column
            self.sort_descending = column != "image"

        index = {
            "overall": 0,
            "image": 1,
            "color": 2,
            "embedding": 3,
            "hash": 4,
        }
        key_index = index[column]

        if column == "image":
            key_fn = lambda row: self.image_by_id[row[1]]["filename"].lower()
        else:
            key_fn = lambda row: row[key_index]

        self.current_results.sort(key=key_fn, reverse=self.sort_descending)
        self._render_results_table()
        self._update_heading_labels()
        self._show_top5(self.current_results, column)

    def _render_results_table(self):
        for item in self.results.get_children():
            self.results.delete(item)

        for overall, image_id, color, embedding, hash_value in self.current_results:
            self.results.insert(
                "",
                "end",
                iid=str(image_id),
                values=(
                    self.image_by_id[image_id]["filename"],
                    f"{overall:.4f}",
                    f"{color:.4f}",
                    f"{embedding:.4f}",
                    f"{hash_value:.4f}",
                ),
            )

    def select_reference(self, reference_id):
        self.selected_id = reference_id

        ref = self.image_by_id[reference_id]

        self.reference_label.config(
            text=f"Referenz: {ref['filename']}"
        )

        self.reference_score.config(
            text=f"{ref['width']} × {ref['height']} px  |  {ref['filepath']}"
        )

        similarity_map = calculate_similarities_for_reference(
            self.repo,
            reference_id,
            self.embedding_index,
            candidate_count=100,
        )

        results = []

        for image_id, scores in similarity_map.items():

            if image_id not in self.image_by_id:
                continue

            color = scores.get("color", 0.0)
            embedding = scores.get("embedding", 0.0)
            hash_value = scores.get("hash", 0.0)

            overall = (
                color +
                embedding +
                hash_value
            ) / 3.0

            results.append(
                (
                    overall,
                    image_id,
                    color,
                    embedding,
                    hash_value,
                )
            )

        self.current_results = results

        self.sort_column = "overall"
        self.sort_descending = True

        self.current_results.sort(
            key=lambda row: row[0],
            reverse=True,
        )

        self._show_top5(
            self.current_results,
            self.sort_column,
        )

        self._render_results_table()
        self._update_heading_labels()

def calculate_histogram_worker(row):
    import time

    start = time.perf_counter()

    try:
        with Image.open(row["filepath"]) as source:
            source.load()

            loaded = time.perf_counter()

            image = source.convert("RGB")

        converted = time.perf_counter()

        histogram = calculate_color_histogram(image)

        calculated = time.perf_counter()

        return (
            row["id"],
            histogram.tobytes(),
            loaded - start,
            converted - loaded,
            calculated - converted,
        )

    except Exception as e:
        print(f"Error processing {row['filepath']}: {e}")
        return row["id"], None, 0, 0, 0

def backfill_color_histograms(repo):

    rows = repo.get_images_missing_color_histogram()
    print(f"Rows missing histogram: {len(rows)}")

    total = len(rows)

    if total == 0:
        print("All color histograms are already cached.")
        return

    print(
        f"Calculating color histograms for {total} existing images..."
    )

    processed = 0

    # Nur die wirklich benötigten Daten an die Worker schicken
    worker_rows = [
        {
            "id": row["id"],
            "filepath": row["filepath"],
        }
        for row in rows
    ]

    with ProcessPoolExecutor(max_workers=8) as executor:

        for (
            image_id,
            histogram,
            load_time,
            convert_time,
            histogram_time,
        ) in executor.map(
            calculate_histogram_worker,
            worker_rows,
        ):

            if histogram is not None:
                repo.db.execute(
                    """
                    UPDATE images
                    SET color_histogram = ?
                    WHERE id = ?
                    """,
                    (
                        histogram,
                        image_id,
                    ),
                    commit=False,
                )

            processed += 1

            if processed % 1000 == 0:
                repo.db.commit()

                print(
                    f"Color histograms: {processed}/{total}"
                )
                print(
                    f"Last image: "
                    f"load={load_time:.3f}s | "
                    f"convert={convert_time:.3f}s | "
                    f"histogram={histogram_time:.3f}s"
                )

    repo.db.commit()

    print(
        f"Finished color histogram backfill: "
        f"{processed}/{total}"
    )

def main():
    db = Database(DB_PATH)
    initialize_database(db)
    repo = ImageRepository(db)

    process_images(repo)

    # Einmaliger Nachlauf für bereits vorhandene Bilder
    backfill_color_histograms(repo)

    repair_similarity_values(db)

    print("Finished importing images.")

    images = repo.get_all_images_for_viewer()
    print(f"Images for viewer: {len(images)}")
    
    embedding_index = EmbeddingIndex(
        "embeddings.index"
    )

    if embedding_index.exists():
        embedding_index.load()
    else:
        print("Building embedding index...")

        index_images = repo.get_all_images_with_embeddings()
        embedding_index.build(index_images)
        embedding_index.save()

    root = tk.Tk()

    SimilarityViewer(
        root,
        repo,
        images,
        embedding_index
    )

    root.mainloop()


if __name__ == "__main__":
    main()
