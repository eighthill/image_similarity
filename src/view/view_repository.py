import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import logging

import imagehash
import cv2
import numpy as np
from PIL import Image, ImageOps
import time
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from config import LOGGER_ACTIVE, IMAGE_FOLDER, CANDIDATE_COUNT
from image_loader.generator import image_generator
from embedding.extractor import extract_embeddings
from embedding.embedding_index import EmbeddingIndex

from similarity.color_similarity import (color_similarity, calculate_color_histogram, color_similarity_from_histograms)
from similarity.embedding_similarity import embedding_similarity
from similarity.hash_similarity import hash_similarity

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
                    #print(f"Skipping image: {filepath} -> {error}")
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

    if LOGGER_ACTIVE:
        logging.info(
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

"""
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
"""

def calculate_similarities_for_reference(
    repo,
    reference_id,
    embedding_index,
    candidate_count=CANDIDATE_COUNT,
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

    ref_embedding = np.frombuffer(reference["embedding"], dtype=np.float32)

    #print(
    #    "index:",
    #    embedding_index.index.ntotal,
    #    embedding_index.index.d,
    #    flush=True,
    #)

    #print(
    #    "embedding:",
    #    ref_embedding.shape,
    #    ref_embedding.dtype,
    #    ref_embedding.flags["C_CONTIGUOUS"],
    #    flush=True,
    #)

    #print("Before FAISS search", flush=True)

    candidates = embedding_index.search(
        ref_embedding,
        k=candidate_count,
    )

    #print(
    #    "After FAISS search:",
    #    len(candidates),
    #    flush=True,
    #)

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
        #print(f"Error processing {row['filepath']}: {e}")
        #return row["id"], None, 0, 0, 0
        pass

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
    
def embedding_builder(repo, index_path):

    embedding_index = EmbeddingIndex(index_path)

    if embedding_index.exists():
        embedding_index.load()
    else:
        print("Building embedding index...")

        index_images = repo.get_all_images_with_embeddings()

        embedding_index.build(index_images)
        embedding_index.save()

    return embedding_index