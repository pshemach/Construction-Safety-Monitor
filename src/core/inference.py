import time
from typing import List
import numpy as np
from ultralytics import YOLO
import torch
from collections import Counter
from ..constant import *
from ..entity.inference_entity import Detection, SceneReport

class SafetyInspector:
    """
    YOLOv9 model and applies safety rules to produce
    structured SceneReport objects.
    """
    def __init__(self, weights: str, conf: float = DEFAULT_CONF,
                 iou: float = DEFAULT_IOU, device: str = None):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def detect_image(self, image_path: str) -> SceneReport:
        to = time.perf_counter()
        
        results = self.model.predict(
            source = image_path,
            conf = self.conf,
            iou = self.iou,
            device = self.device,
            verbose = True
        )
        
        elapsed_ms = (time.perf_counter() - to) * 1000
        
        return self._build_report(image_path, results[0], elapsed_ms)
    
    def inspect_frame(self, frame: np.ndarray) -> SceneReport:
        t0      = time.perf_counter()
        results = self.model.predict(
            source  = frame,
            conf    = self.conf,
            iou     = self.iou,
            device  = self.device,
            verbose = False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return self._build_report("frame", results[0], elapsed_ms)  
    
    def _build_report(self, source: str, result, elapsed_ms: float) -> SceneReport:
        detections: List[Detection] = []
        violations: List[Detection] = []
        
        names = self.model.names  # {int: str}
        
        for box in result.boxes:
            label = names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            is_viol = label in VIOLATION_CLASSES
            
            det = Detection(label=label, confidence=conf,
                            bbox=(x1, y1, x2, y2), is_violation=is_viol)
            
            detections.append(det)
            if is_viol:
                violations.append(det)
                
            all_confs = [d.confidence for d in detections] if detections else [0.0]
            base_conf = float(np.mean(all_confs))
            
            # each violation reduces confidence in the SAFE verdict
            scene_conf = self._compute_scene_confidence(base_conf, violations, detections)
            
            verdict = "UNSAFE" if violations else "SAFE"
            alert   = self._generate_alert(violations, scene_conf)
            
            return SceneReport(
            image_path       = str(source),
            verdict          = verdict,
            scene_confidence = scene_conf,
            violations       = violations,
            detections       = detections,
            inference_ms     = elapsed_ms,
            alert_message    = alert
        )
    
    @staticmethod
    def _compute_scene_confidence(base_conf: float, 
                                  violations: List[Detection],
                                  all_detections: List[Detection]) -> float:
        if not all_detections:
            return 0.5   # too uncertain — no detections at all
        
        # Average confidence of the detections driving the verdict
        driving = violations if violations else [d for d in all_detections
                                                  if d.label in COMPLIANT_CLASSES]
        
        if not driving:
            driving = all_detections
            
        verdict_conf = float(np.mean([d.confidence for d in driving]))
        
        return round(min(max(verdict_conf, 0.0), 1.0), 4)
    
    @staticmethod
    def _generate_alert(violations: List[Detection], confidence: float) -> str:
        if not violations:
            return f"Scene is SAFE — no PPE violations detected ({confidence*100:.0f}% confidence)."
        
        pct = confidence * 100
        n = len(violations)
        noun = "violation" if n == 1 else "violations"
        
        counts = Counter(v.label for v in violations)
        details = []
        for label, cnt in counts.items():
            rule_desc = VIOLATION_CLASSES.get(label, label)
            prefix = f"{cnt}× " if cnt > 1 else ""
            details.append(f"  • {prefix}{label}: {rule_desc}")
            
        lines = [
            f"⚠  UNSAFE — {n} PPE {noun} detected ({pct:.0f}% confidence):",
            *details,
        ]
        return "\n".join(lines)