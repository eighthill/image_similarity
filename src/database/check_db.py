import sqlite3

conn = sqlite3.connect("images.db")

cursor = conn.cursor()

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
""")

for row in cursor.fetchall():
    print(row)