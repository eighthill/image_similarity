class ImageRepository:

    def __init__(self, db):
        self.db = db

    def insert_image(
        self,
        filename,
        filepath,
        width,
        height,
        embedding,
        hash_value,
        color_histogram
    ):
        self.db.execute(
            """
            INSERT OR IGNORE INTO images
            (
                filename,
                filepath,
                width,
                height,
                embedding,
                hash_value,
                color_histogram
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                filepath,
                width,
                height,
                embedding,
                hash_value,
                color_histogram
            ),
            commit=False
        )

    def get_images_missing_color_histogram(self):
        cursor = self.db.execute(
            """
            SELECT id, filepath
            FROM images
            WHERE color_histogram IS NULL
            """,
            commit=False
        )

        return cursor.fetchall()

    def get_image_by_filepath(self, filepath):
        cursor = self.db.execute(
            "SELECT * FROM images WHERE filepath = ?",
            (filepath,)
        )
        return cursor.fetchone()
    
    def get_image_by_id(self, image_id):
            cursor = self.db.execute(
                """
                SELECT *
                FROM images
                WHERE id = ?
                """,
                (image_id,),
                commit=False,
            )
    
            return cursor.fetchone()
    
    def get_existing_image_paths(self):
        cursor = self.db.execute(
            """
            SELECT id, filepath,
                embedding IS NOT NULL AS has_embedding,
                hash_value IS NOT NULL AS has_hash,
                color_histogram IS NOT NULL AS has_color_histogram
            FROM images
            """,
            commit=False
        )
        return cursor.fetchall()

    def get_all_images(self):
        cursor = self.db.execute(
            "SELECT * FROM images"
        )
        return cursor.fetchall()

    def update_image_features(self, image_id, embedding, hash_value, color_histogram):
        self.db.execute(
            """
            UPDATE images
            SET embedding = ?, hash_value = ?, color_histogram = ?
            WHERE id = ?
            """,
            (embedding, hash_value, color_histogram, image_id),
            commit=False
        )

    def get_existing_similarity_pairs(self, table_name):
        if table_name not in {
            "color_similarities",
            "embedding_similarities",
            "hash_similarities",
        }:
            raise ValueError(f"Invalid similarity table: {table_name}")

        cursor = self.db.execute(
            f"SELECT id1, id2 FROM {table_name}"
        )
        return {(min(row["id1"], row["id2"]), max(row["id1"], row["id2"])) for row in cursor.fetchall()}

    def insert_color_similarity(self, id1, id2, similarity):
        self.db.execute(
            """
            INSERT OR IGNORE INTO
            color_similarities
            VALUES (?, ?, ?)
            """,
            (id1, id2, float(similarity)),
            commit=False
        )

    def insert_embedding_similarity(self, id1, id2, similarity):
        self.db.execute(
            """
            INSERT OR IGNORE INTO
            embedding_similarities
            VALUES (?, ?, ?)
            """,
            (id1, id2, float(similarity)),
            commit=False
        )

    def insert_hash_similarity(self, id1, id2, similarity):
        self.db.execute(
            """
            INSERT OR IGNORE INTO
            hash_similarities
            VALUES (?, ?, ?)
            """,
            (id1, id2, float(similarity)),
            commit=False
        )
    
    def get_all_images_for_viewer(self):
        cursor = self.db.execute(
            """
            SELECT
                id,
                filename,
                filepath,
                width,
                height
            FROM images
            """,
            commit=False
        )

        return cursor.fetchall()


    def get_all_images_with_embeddings(self):
        cursor = self.db.execute(
            """
            SELECT
                id,
                embedding
            FROM images
            WHERE embedding IS NOT NULL
            """,
            commit=False
        )

        return cursor.fetchall()