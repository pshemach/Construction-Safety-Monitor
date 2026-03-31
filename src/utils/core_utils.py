import cv2
import csv
import json
import numpy as np
from pathlib import Path
from typing import List
from ..entity.inference_entity import SceneReport
from ..core.inference import SafetyInspector
from ..constant import *


def draw_report(image: np.ndarray, report: SceneReport) -> np.ndarray:
    """Draw bounding boxes and verdict banner onto an image."""
    img = image.copy()
    h, w = img.shape[:2]

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