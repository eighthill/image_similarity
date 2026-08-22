import sqlite3

sql_statements = [
    """CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL UNIQUE,
        width INTEGER,
        height INTEGER,
        photographer TEXT,
        embedding BLOB,
        hash_value TEXT
    );"""
    ,
    """CREATE TABLE IF NOT EXISTS color_similarities (
        id1 INTEGER,
        id2 INTEGER,
        similarity REAL,
        PRIMARY KEY (id1, id2)
    );"""
    ,
    """CREATE TABLE IF NOT EXISTS embedding_similarities (
        id1 INTEGER,
        id2 INTEGER,
        similarity REAL,
        PRIMARY KEY (id1, id2)
    );"""
    ,
    """CREATE TABLE IF NOT EXISTS hash_similarities (
        id1 INTEGER,
        id2 INTEGER,
        similarity REAL,
        PRIMARY KEY (id1, id2)
    );"""
]

# create a database connection
try:
    with sqlite3.connect('images.db') as conn:
        # create a cursor
        cursor = conn.cursor()

        # execute statements
        for statement in sql_statements:
            cursor.execute(statement)

        # commit the changes
        conn.commit()

        print("Tables created successfully.")
except sqlite3.OperationalError as e:
    print("Failed to create tables:", e)

