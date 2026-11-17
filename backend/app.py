
import os
import cv2
import uuid
import sqlite3

from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLO

app = FastAPI(title="Pothole Detection API")

PROJECT = "/content/pothole-severity-mapping"

MODEL_PATH = f"{PROJECT}/runs/pothole_yolov8n/weights/best.pt"
DB_PATH = f"{PROJECT}/pothole_database.db"
OUTPUT_DIR = f"{PROJECT}/outputs/api"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


def initialize_database():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            confidence REAL,
            area_percentage REAL,
            severity TEXT,
            latitude REAL,
            longitude REAL
        )
    """)

    conn.commit()
    conn.close()


initialize_database()


@app.get("/")
def home():
    return {
        "message": "Pothole Detection API is running!"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    latitude: float = Form(None),
    longitude: float = Form(None)
):

    image_name = f"{uuid.uuid4().hex}_{file.filename}"
    image_path = os.path.join(OUTPUT_DIR, image_name)

    contents = await file.read()

    with open(image_path, "wb") as f:
        f.write(contents)

    image = cv2.imread(image_path)

    if image is None:
        return {"error": "Invalid image"}

    height, width = image.shape[:2]

    results = model.predict(
        image,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    detections = []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for box in result.boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        confidence = float(box.conf[0])

        area = (
            (x2 - x1) * (y2 - y1)
            / (width * height)
        ) * 100

        if area < 1:
            severity = "LOW"
        elif area < 5:
            severity = "MEDIUM"
        else:
            severity = "HIGH"

        cursor.execute("""
            INSERT INTO api_detections
            (image_name, confidence, area_percentage,
             severity, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            image_name,
            confidence,
            area,
            severity,
            latitude,
            longitude
        ))

        detections.append({
            "confidence": round(confidence, 3),
            "area_percentage": round(float(area), 3),
            "severity": severity,
            "coordinates": [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2), 2),
                round(float(y2), 2)
            ]
        })

    conn.commit()
    conn.close()

    return {
        "image": image_name,
        "total_potholes": len(detections),
        "detections": detections,
        "status": "success"
    }
