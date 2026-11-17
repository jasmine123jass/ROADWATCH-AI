
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "pothole_database.db"


def save_detection(image_name, confidence, area_percentage,
                   severity, latitude=None, longitude=None):

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO api_detections
            (image_name, confidence, area_percentage,
             severity, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            image_name,
            confidence,
            area_percentage,
            severity,
            latitude,
            longitude
        ))

        connection.commit()
        return cursor.lastrowid

    finally:
        connection.close()


def get_all_detections():

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT * FROM api_detections
            ORDER BY id DESC
        """)

        return [dict(row) for row in cursor.fetchall()]

    finally:
        connection.close()
