import faiss
import numpy as np
from pathlib import Path


class EmbeddingIndex:

    def __init__(self, index_path):
        self.index_path = Path(index_path)
        self.ids_path = self.index_path.with_suffix(".ids.npy")

        self.index = None
        self.image_ids = None

    def build(self, images):

        embeddings = []
        image_ids = []

        for image in images:

            if image["embedding"] is None:
                continue

            embedding = np.frombuffer(
                image["embedding"],
                dtype=np.float32,
            )

            norm = np.linalg.norm(embedding)

            if norm == 0:
                continue

            embedding = embedding / norm

            embeddings.append(embedding)
            image_ids.append(image["id"])

        if not embeddings:
            raise RuntimeError(
                "Keine Embeddings für den Index gefunden."
            )

        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        index = faiss.IndexFlatIP(
            matrix.shape[1]
        )

        index.add(matrix)

        self.index = index

        self.image_ids = np.asarray(
            image_ids,
            dtype=np.int64,
        )

    def save(self):

        if self.index is None:
            raise RuntimeError(
                "Index wurde noch nicht erstellt."
            )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        np.save(
            self.ids_path,
            self.image_ids,
        )

        print(
            f"Embedding index saved: "
            f"{len(self.image_ids)} images"
        )

    def load(self):

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.image_ids = np.load(
            self.ids_path,
        )

        print(
            f"Embedding index loaded: "
            f"{len(self.image_ids)} images"
        )

    def exists(self):

        return (
            self.index_path.exists()
            and self.ids_path.exists()
        )

    def search(self, embedding, k=100):

        if self.index is None:
            raise RuntimeError(
                "Embedding index ist nicht geladen."
            )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return []

        embedding = embedding / norm

        # Nicht mehr Ergebnisse anfordern,
        # als tatsächlich im Index vorhanden sind.
        k = min(
            k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            embedding.reshape(1, -1),
            k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            image_id = int(
                self.image_ids[index]
            )

            results.append(
                (
                    image_id,
                    float(score),
                )
            )

        return results