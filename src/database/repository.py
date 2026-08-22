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
        hash_value
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
                hash_value
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                filepath,
                width,
                height,
                embedding,
                hash_value
            )
        )

    def get_all_images(self):
        cursor = self.db.execute(
            "SELECT * FROM images"
        )
        return cursor.fetchall()
    
    def insert_color_similarity(self, id1, id2, similarity):
        self.db.execute(
            """
            INSERT OR REPLACE INTO
            color_similarities
            VALUES (?, ?, ?)
            """,
            (id1, id2, similarity)
        )


    def insert_embedding_similarity(self, id1, id2, similarity):
        self.db.execute(
            """
            INSERT OR REPLACE INTO
            embedding_similarities
            VALUES (?, ?, ?)
            """,
            (id1, id2, similarity)
        )


    def insert_hash_similarity(self, id1, id2, similarity):
        self.db.execute(
            """
            INSERT OR REPLACE INTO
            hash_similarities
            VALUES (?, ?, ?)
            """,
            (id1, id2, similarity)
        )