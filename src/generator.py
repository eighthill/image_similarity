import sqlite3
import cv2

def image_generator(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, file_path
        FROM images
    """)

    for image_id, file_path in cursor:
        yield image_id, cv2.imread(file_path)

    conn.close()