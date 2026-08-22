import sqlite3


class Database:

    def __init__(self, db_path):
        print(db_path)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def execute(self, query, params=(), commit=True):
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        if commit:
            self.conn.commit()
        return cursor

    def commit(self):
        self.conn.commit()
