
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.pothole_detector import PotholeDetector
from src.severity_estimator import estimate_severity
from src.database import save_detection, get_all_detections

app = FastAPI(
    title="Pothole Detection and Severity Mapping API",
    description="API for detecting potholes in uploaded road images.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

detector = PotholeDetector()


@app.get("/health")
def health_check():
    return {
        "status": "running",
        "message": "Pothole Detection API is working"
    }


@app.post("/predict")
async def predict_potholes(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    extension = Path(file.filename or "").suffix.lower()

    if extension not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    image_name = f"{uuid.uuid4().hex}{extension}"
    image_path = UPLOAD_DIR / image_name

    try:
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        detections, annotated_image = detector.detect(str(image_path))

        results = []

        for detection in detections:
            severity = estimate_severity(
                detection["area_percentage"]
            )

            record_id = save_detection(
                image_name=file.filename or image_name,
                confidence=detection["confidence"],
                area_percentage=detection["area_percentage"],
                severity=severity
            )

            results.append({
                "database_id": record_id,
                "confidence": detection["confidence"],
                "area_percentage": detection["area_percentage"],
                "severity": severity,
                "bbox": detection["bbox"]
            })

        return {
            "image": file.filename,
            "total_potholes": len(results),
            "detections": results
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    finally:
        await file.close()


@app.get("/detections")
def list_detections():
    return {
        "total_records": len(get_all_detections()),
        "records": get_all_detections()
    }
