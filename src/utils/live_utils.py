import cv2
from .images_utils import draw_report
from ..core.inference import SafetyInspector
from ..constant import *

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