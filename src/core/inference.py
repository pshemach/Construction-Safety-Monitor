import argparse
import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import Counter
from ..constant import *

# ── Data classes ──────────────────────────────────────────────────────────────
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

# ── Annotated image rendering ─────────────────────────────────────────────────
def draw_report(image: np.ndarray, report: SceneReport) -> np.ndarray:
    """Draw bounding boxes and verdict banner onto an image."""
    img = image.copy()
    h, w = img.shape[:2]

    # Bounding boxes
    for det in report.detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        colour = COLOUR_MAP.get(det.label, (200, 200, 200))
        thickness = 3 if det.is_violation else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

        label_text = f"{det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(img, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Verdict banner at top
    banner_colour = (0, 0, 200) if report.verdict == "UNSAFE" else (0, 160, 60)
    banner_h = 44
    cv2.rectangle(img, (0, 0), (w, banner_h), banner_colour, -1)

    banner_text = (f"{'⚠ ' if report.verdict == 'UNSAFE' else '✓ '}"
                   f"{report.verdict}  |  "
                   f"{len(report.violations)} violation(s)  |  "
                   f"{report.scene_confidence*100:.0f}% conf  |  "
                   f"{report.inference_ms:.0f}ms")
    cv2.putText(img, banner_text, (10, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return img


# ── Batch processing ──────────────────────────────────────────────────────────
def run_batch(inspector: SafetyInspector, source_dir: Path,
              output_dir: Path) -> List[SceneReport]:
    """Process all images in source_dir, save annotated images + JSON reports."""
    ann_dir    = output_dir / "annotated"
    report_dir = output_dir / "reports"
    ann_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(source_dir.glob("*.[jp][pn]g")) + \
                  sorted(source_dir.glob("*.jpeg"))

    if not image_paths:
        print(f"[!] No images found in {source_dir}")
        return []

    reports: List[SceneReport] = []
    safe_count = unsafe_count = 0

    print(f"\n[→] Processing {len(image_paths)} images …\n")

    for i, img_path in enumerate(image_paths, 1):
        report = inspector.inspect_image(str(img_path))
        reports.append(report)

        # Save annotated image
        frame = cv2.imread(str(img_path))
        if frame is not None:
            annotated = draw_report(frame, report)
            out_img = ann_dir / img_path.name
            cv2.imwrite(str(out_img), annotated)

        # Save JSON report
        json_path = report_dir / (img_path.stem + ".json")
        json_path.write_text(json.dumps(report.to_dict(), indent=2))

        status = "🔴 UNSAFE" if report.verdict == "UNSAFE" else "🟢 SAFE"
        print(f"  [{i:4d}/{len(image_paths)}]  {status}  {img_path.name}"
              f"  ({report.inference_ms:.0f}ms)  "
              f"{len(report.violations)} violation(s)")

        if report.verdict == "UNSAFE":
            print(f"          {report.alert_message}")
            unsafe_count += 1
        else:
            safe_count += 1

    # ── Summary CSV ──────────────────────────────────────────────────────────
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "verdict", "confidence", "violations", "alert", "inference_ms"])
        writer.writeheader()
        for r in reports:
            writer.writerow({
                "image":        Path(r.image_path).name,
                "verdict":      r.verdict,
                "confidence":   r.scene_confidence,
                "violations":   len(r.violations),
                "alert":        r.alert_message.replace("\n", " | "),
                "inference_ms": r.inference_ms,
            })

    # ── Final summary ────────────────────────────────────────────────────────
    total = safe_count + unsafe_count
    print(f"\n{'═'*60}")
    print(f"  Results: {total} images processed")
    print(f"  🟢 SAFE:   {safe_count}  ({safe_count/total*100:.1f}%)")
    print(f"  🔴 UNSAFE: {unsafe_count}  ({unsafe_count/total*100:.1f}%)")
    print(f"  Annotated images → {ann_dir}")
    print(f"  JSON reports     → {report_dir}")
    print(f"  Summary CSV      → {csv_path}")
    print("═"*60)

    return reports


# ── Live video / webcam ───────────────────────────────────────────────────────
def run_live(inspector: SafetyInspector, source):
    """Real-time inference on webcam or video file."""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        print(f"[!] Cannot open source: {source}")
        return

    print("[→] Running live inference — press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        report   = inspector.inspect_frame(frame)
        annotated = draw_report(frame, report)
        cv2.imshow("Construction Safety Monitor", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()