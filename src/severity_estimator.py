
def estimate_severity(area_percentage):

    # Initial image-area heuristic.
    # These thresholds require real-world calibration.

    if area_percentage < 1:
        return "LOW"

    elif area_percentage < 5:
        return "MEDIUM"

    else:
        return "HIGH"
