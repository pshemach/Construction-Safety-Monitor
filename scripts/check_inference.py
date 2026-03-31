import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cv2
from src.core.inference import SafetyInspector
from src.constant import *
from src.utils.images_utils import *
from src.utils.videos_utils import *
from src.utils.live_utils import run_live

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="data/model.pt", help="Path to model.pt")
    p.add_argument("--source", required=True, help="Video file, image, image folder, or webcam index")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--skip-frames", type=int, default=2, help="Process every N+1 frames. 0=every, 2=every 3rd (3x faster)")
    p.add_argument("--no-display", action="store_true", help="Suppress cv2.imshow — required in Colab/headless")
    p.add_argument("--live", action="store_true")
    p.add_argument("--device", default=None,  help="'cuda', 'cpu', or device index")
    return p.parse_args()


def main():
    args      = parse_args()
    inspector = SafetyInspector(args.weights, args.conf, args.iou, args.device)
    source    = Path(args.source)

    if args.live or (not source.exists() and str(args.source).isdigit()):
        run_live(inspector, args.source)
    elif source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
        run_video(inspector, source, args.output_dir,
                  no_display=args.no_display, skip_frames=args.skip_frames)
    elif source.is_dir():
        run_batch(inspector, source, args.output_dir)
    elif source.is_file():
        report = inspector.detect_image(str(source))
        print("\n" + report.alert_message)
        frame = cv2.imread(str(source))
        annotated = draw_report(frame, report)
        out_path = args.output_dir / "annotated" / source.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
    else:
        print(f"Source not found: {args.source}")


if __name__ == "__main__":
    main()
