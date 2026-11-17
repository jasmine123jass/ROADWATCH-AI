
from src.pothole_detector import PotholeDetector
from src.severity_estimator import estimate_severity
from src.database import save_detection

import cv2
from pathlib import Path

IMAGE = "data/processed/potholes/images/test/img-10.jpg"

detector = PotholeDetector()
detections, annotated_image = detector.detect(IMAGE)

print("\n===== POTHOLE DETECTION RESULTS =====")

for i, detection in enumerate(detections, 1):

    severity = estimate_severity(
        detection["area_percentage"]
    )

    record_id = save_detection(
        image_name=Path(IMAGE).name,
        confidence=detection["confidence"],
        area_percentage=detection["area_percentage"],
        severity=severity
    )

    print(f"\nPothole {i}")
    print("Database ID:", record_id)
    print("Confidence:", detection["confidence"])
    print("Area:", detection["area_percentage"], "%")
    print("Severity:", severity)
    print("Saved to database successfully!")

if detections:
    output_path = "outputs/local_detection.jpg"
    cv2.imwrite(output_path, annotated_image)
    print("\nOutput:", output_path)
else:
    print("\nNo potholes detected.")
