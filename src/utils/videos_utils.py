import cv2
import time
import json
import numpy as np
import csv
from collections import Counter
from typing import List, Optional
from pathlib import Path
from src.entity.inference_entity import FrameReport
from src.core.inference import SafetyInspector
from src.constant import *

def draw_frame(frame: np.ndarray, report: FrameReport,
               show_timestamp: bool = True) -> np.ndarray:
    img   = frame.copy()
    h, w  = img.shape[:2]

    for det in report.detections:
        x1,y1,x2,y2 = (int(v) for v in det.bbox)
        col = COLOUR_MAP.get(det.label, (200,200,200))
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 3 if det.is_violation else 2)
        txt = f"{det.label} {det.confidence:.2f}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), col, -1)
        cv2.putText(img, txt, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    banner_col = (0,0,180) if report.verdict == "UNSAFE" else (0,140,50)
    cv2.rectangle(img, (0,0), (w,40), banner_col, -1)
    banner = (f"{'[UNSAFE]' if report.verdict == 'UNSAFE' else '[SAFE]'}"
              f"  {len(report.violations)} violation(s)"
              f"  |  {report.scene_confidence*100:.0f}% conf"
              f"  |  {report.inference_ms:.0f}ms")
    cv2.putText(img, banner, (10,27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

    if show_timestamp:
        mins = int(report.timestamp_sec) // 60
        secs = int(report.timestamp_sec) % 60
        cv2.putText(img, f"{mins:02d}:{secs:02d}", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)

    if report.violations:
        vio_cnt = Counter(v.label for v in report.violations)
        y_pos   = h - 10 - (len(vio_cnt)-1)*22
        for label, cnt in vio_cnt.items():
            txt = f"{'x'+str(cnt)+' ' if cnt>1 else ''}{label}"
            (tw,_),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(img, txt, (w-tw-10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,255), 2, cv2.LINE_AA)
            y_pos += 22

    return img


def run_video(inspector: SafetyInspector,
              video_path: Path,
              output_dir: Path,
              no_display: bool = False,
              skip_frames: int = 0) -> None:
    """
    Process a video file and save:
      - annotated MP4
      - per-second timeline CSV

    skip_frames=0  -> process every frame   (accurate, slower)
    skip_frames=2  -> process every 3rd frame (3x faster)
    skip_frames=4  -> process every 5th frame (5x faster)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Cannot open video: {video_path}")
        print("    Check the file path and that cv2 supports the codec.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    print(f"\n{'='*60}")
    print(f"  Video    : {video_path.name}")
    print(f"  Size     : {width}x{height}  {fps:.1f}fps  {duration_sec:.1f}s")
    print(f"  Frames   : {total_frames}  (processing every {skip_frames+1})")
    print(f"{'='*60}\n")

    # Output MP4 writer
    out_path = output_dir / f"annotated_{stem}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_reports: List[FrameReport] = []
    last_report:   Optional[FrameReport] = None
    safe_count = unsafe_count = 0
    frame_idx  = 0
    t_start    = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        if frame_idx % (skip_frames + 1) == 0:
            report      = inspector.inspect_frame(frame, frame_idx, timestamp)
            last_report = report
            frame_reports.append(report)
            if report.verdict == "UNSAFE":
                unsafe_count += 1
            else:
                safe_count += 1
        else:
            # Reuse last known report for skipped frames
            if last_report:
                report = FrameReport(
                    frame_idx        = frame_idx,
                    timestamp_sec    = timestamp,
                    verdict          = last_report.verdict,
                    scene_confidence = last_report.scene_confidence,
                    violations       = last_report.violations,
                    detections       = last_report.detections,
                    inference_ms     = 0.0,
                )
            else:
                report = FrameReport(frame_idx, timestamp, "SAFE", 0.5)

        annotated = draw_frame(frame, report)
        writer.write(annotated)

        if frame_idx % 30 == 0 and frame_idx > 0:
            elapsed  = time.perf_counter() - t_start
            pct      = frame_idx / max(total_frames, 1) * 100
            fps_proc = frame_idx / max(elapsed, 0.001)
            eta      = (total_frames - frame_idx) / max(fps_proc, 0.1)
            icon     = "UNSAFE" if (last_report and last_report.verdict == "UNSAFE") else "SAFE  "
            print(f"  Frame {frame_idx:5d}/{total_frames} ({pct:5.1f}%)"
                  f"  {fps_proc:5.1f} fps  ETA {eta:.0f}s  {icon}")

        frame_idx += 1

    cap.release()
    writer.release()
    elapsed_total = time.perf_counter() - t_start

    print(f"\n[OK] Annotated video : {out_path}")

    # Per-second timeline CSV
    timeline_path = output_dir / f"timeline_{stem}.csv"
    _write_timeline(frame_reports, timeline_path)
    print(f"[OK] Timeline CSV    : {timeline_path}")

    print(f"\n{'='*60}")
    total_proc = safe_count + unsafe_count
    print(f"  SAFE   : {safe_count} frames  ({safe_count/max(total_proc,1)*100:.1f}%)")
    print(f"  UNSAFE : {unsafe_count} frames  ({unsafe_count/max(total_proc,1)*100:.1f}%)")
    print(f"  Output : {output_dir}/")
    print("="*60)


def _write_timeline(reports: List[FrameReport], path: Path) -> None:
    from collections import defaultdict
    by_sec: dict = defaultdict(list)
    for r in reports:
        by_sec[int(r.timestamp_sec)].append(r)

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "timestamp", "verdict",
                    "unsafe_frames", "total_frames",
                    "violation_classes", "max_violations"])
        for sec in sorted(by_sec):
            rows     = by_sec[sec]
            unsafe   = [r for r in rows if r.verdict == "UNSAFE"]
            viols    = sorted(set(v.label for r in rows for v in r.violations))
            max_v    = max((len(r.violations) for r in rows), default=0)
            verdict  = "UNSAFE" if unsafe else "SAFE"
            m, s     = sec//60, sec%60
            w.writerow([sec, f"{m:02d}:{s:02d}", verdict,
                        len(unsafe), len(rows), ", ".join(viols), max_v])