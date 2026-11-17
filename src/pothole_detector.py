
from pathlib import Path
from ultralytics import YOLO
import cv2


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "runs" / "pothole_yolov8n" / "weights" / "best.pt"


class PotholeDetector:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        self.model = YOLO(str(MODEL_PATH))

    def detect(self, image_path, confidence=0.25):

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Could not read input image")

        height, width = image.shape[:2]

        results = self.model.predict(
            source=image,
            conf=confidence,
            imgsz=640,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    float, box.xyxy[0].tolist()
                )

                conf = float(box.conf[0])

                bbox_area = max(0, x2-x1) * max(0, y2-y1)
                area_percentage = (
                    bbox_area / (width * height)
                ) * 100

                detections.append({
                    "confidence": round(conf, 4),
                    "bbox": [
                        round(x1), round(y1),
                        round(x2), round(y2)
                    ],
                    "area_percentage": round(area_percentage, 3)
                })

        return detections, results[0].plot()
