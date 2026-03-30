# Safety violations
VIOLATION_CLASSES = {
    "NO-Hardhat":     "Worker detected without a hard hat",
    "NO-Safety Vest": "Worker detected without a high-visibility safety vest",
    "NO-Mask":        "Worker detected without a face mask",
}

# Classes that represent compliant PPE
COMPLIANT_CLASSES = {"Hardhat", "Safety Vest", "Mask", "Gloves"}

# Colour palette for bounding boxes (BGR for OpenCV)
COLOUR_MAP = {
    "Hardhat":      (0,  200, 100),   # green
    "Safety Vest":  (0,  180, 255),   # amber-ish
    "Mask":         (255, 180,  0),   # blue-ish
    "Person":       (200, 200, 200),  # grey
    "NO-Hardhat":   (0,   0,  220),   # red
    "NO-Safety Vest":(0,  0,  200),   # dark red
    "NO-Mask":      (60,  0,  200),   # purple-red
    "Gloves":  (0, 220, 255)          # yellow
}

# Model parameters
DEFAULT_CONF = 0.40   # minimum confidence threshold
DEFAULT_IOU  = 0.45   # NMS IoU threshold

# Video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}