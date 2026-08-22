from pathlib import Path
import imagehash
import cv2
import numpy as np

from PIL import Image

from database.database import Database
from database.repository import ImageRepository
from image_loader.generator import image_generator
from embedding.extractor import extract_embedding

from similarity.color_similarity import color_similarity
from similarity.embedding_similarity import embedding_similarity
from similarity.hash_similarity import hash_similarity


DB_PATH = "images.db"
IMAGE_FOLDER = "/Volumes/Extreme SSD/data/image_data/DAISY24"


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
            hash_value TEXT
        )
    """)


def process_images(repo):

    for image, filepath in image_generator(IMAGE_FOLDER):

        try:

            print(f"Processing: {filepath}")

            filename = Path(filepath).name

            width, height = image.size

            embedding = extract_embedding(image)

            hash_value = str(
                imagehash.phash(image)
            )

            repo.insert_image(
                filename=filename,
                filepath=filepath,
                width=width,
                height=height,
                embedding=embedding.tobytes(),
                hash_value=hash_value
            )

        except Exception as e:

            print(
                f"Error processing {filepath}: {e}"
            )

        finally:

            image.close()

def calculate_similarities(repo):
    images = repo.get_all_images()

    for i in range(len(images)):

        for j in range(i + 1, len(images)):

            img1 = images[i]
            img2 = images[j]

            print(
                f"\n{img1['filename']} <-> {img2['filename']}"
            )

            image1 = cv2.imread(
                img1["filepath"]
            )

            image2 = cv2.imread(
                img2["filepath"]
            )

            color_sim = color_similarity(
                image1,
                image2
            )

            e1 = np.frombuffer(
                img1["embedding"],
                dtype=np.float32
            )

            e2 = np.frombuffer(
                img2["embedding"],
                dtype=np.float32
            )

            embedding_sim = embedding_similarity(
                e1,
                e2
            )

            h1 = imagehash.hex_to_hash(
                img1["hash_value"]
            )

            h2 = imagehash.hex_to_hash(
                img2["hash_value"]
            )

            hash_sim = hash_similarity(
                h1,
                h2
            )

            print(
                f"Color: {color_sim:.4f}"
            )

            print(
                f"Embedding: {embedding_sim:.4f}"
            )

            print(
                f"Hash: {hash_sim:.4f}"
            )

            repo.insert_color_similarity(
                img1["id"],
                img2["id"],
                color_sim
            )

            repo.insert_embedding_similarity(
                img1["id"],
                img2["id"],
                embedding_sim
            )

            repo.insert_hash_similarity(
                img1["id"],
                img2["id"],
                hash_sim
            )

def main():

    db = Database(DB_PATH)

    initialize_database(db)

    repo = ImageRepository(db)

    process_images(repo)
    
    print("Calculating similarities...")

    calculate_similarities(repo)

    print("Finished importing images.")

    images = repo.get_all_images()

    for i in range(len(images)):
        for j in range(i + 1, len(images)):

            img1 = images[i]
            img2 = images[j]

            print(
                f"{img1['filename']} <-> {img2['filename']}"
            )

if __name__ == "__main__":
    main()