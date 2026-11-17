
import sqlite3

db = sqlite3.connect("pothole_database.db")
cursor = db.cursor()

print("===== DATABASE TABLES =====")

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
""")

tables = cursor.fetchall()

for table in tables:
    name = table[0]
    print(f"\nTable: {name}")

    cursor.execute(f'PRAGMA table_info("{name}")')

    for column in cursor.fetchall():
        print(column)

db.close()
