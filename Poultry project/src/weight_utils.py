def estimate_weight_index(area: float, min_area: float = 2000, max_area: float = 22000):
    """
    Converts bounding box pixel area → normalized 0–1 range weight index.
    Output:
        0 = smallest bird detected
        1 = largest bird detected
    """

    if area < min_area:
        return 0.00
    if area > max_area:
        return 1.00

    norm = (area - min_area) / (max_area - min_area)
    return round(norm, 3)
