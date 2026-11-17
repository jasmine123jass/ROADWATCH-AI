
from src.database import get_all_detections

records = get_all_detections()

print("\n===== STORED POTHOLE RECORDS =====")

for record in records:
    print(record)

print("\nTotal records:", len(records))
