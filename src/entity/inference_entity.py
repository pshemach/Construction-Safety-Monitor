from dataclasses import dataclass, field, asdict
from typing import Tuple, List

@dataclass
class Detection:
    label:      str
    confidence: float
    bbox:       Tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels)
    is_violation: bool = False

    def to_dict(self):
        return {
            "label":        self.label,
            "confidence":   round(self.confidence, 4),
            "bbox":         [round(v, 1) for v in self.bbox],
            "is_violation": self.is_violation,
        }


@dataclass
class SceneReport:
    image_path:      str
    verdict:         str                # "SAFE" | "UNSAFE"
    scene_confidence: float             # 0–1
    violations:      List[Detection]   = field(default_factory=list)
    detections:      List[Detection]   = field(default_factory=list)
    inference_ms:    float             = 0.0
    alert_message:   str               = ""

    def to_dict(self):
        return {
            "image":            self.image_path,
            "verdict":          self.verdict,
            "scene_confidence": round(self.scene_confidence, 4),
            "alert":            self.alert_message,
            "violations":       [v.to_dict() for v in self.violations],
            "all_detections":   [d.to_dict() for d in self.detections],
            "inference_ms":     round(self.inference_ms, 2),
        }