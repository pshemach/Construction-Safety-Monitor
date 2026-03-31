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
        
@dataclass
class FrameReport:
    frame_idx:        int
    timestamp_sec:    float
    verdict:          str
    scene_confidence: float
    violations:       List[Detection] = field(default_factory=list)
    detections:       List[Detection] = field(default_factory=list)
    inference_ms:     float = 0.0

    def to_dict(self):
        return {
            "frame":      self.frame_idx,
            "time_sec":   round(self.timestamp_sec, 3),
            "verdict":    self.verdict,
            "confidence": round(self.scene_confidence, 4),
            "violations": [v.to_dict() for v in self.violations],
            "detections": [d.to_dict() for d in self.detections],
            "ms":         round(self.inference_ms, 2),
        }