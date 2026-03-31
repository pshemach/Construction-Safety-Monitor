import argparse
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from src.core.inference import SafetyInspector
from src.utils.images_utils import draw_report

_inspector: SafetyInspector = None # loaded once at startup

def load_inspector(weights: str, conf: float, device: str):
    global _inspector
    _inspector = SafetyInspector(weights=weights, conf=conf, device=device)
    print(f"Model ready — launch Gradio UI")


def predict(pil_image: Image.Image):
    """Prediction function"""
    if pil_image is None:
        return None, "—", "Upload an image to get started.", "{}"

    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    report    = _inspector.detect_image(frame)
    annotated = draw_report(frame, report)

    # Convert back to PIL for Gradio
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    output_pil    = Image.fromarray(annotated_rgb)

    # Verdict badge text
    if report.verdict == "UNSAFE":
        verdict_md = f"## 🔴 UNSAFE  ({report.scene_confidence*100:.0f}% confidence)"
    else:
        verdict_md = f"## 🟢 SAFE  ({report.scene_confidence*100:.0f}% confidence)"

    alert_text = report.alert_message
    json_text  = __import__("json").dumps(report.to_dict(), indent=2)

    return output_pil, verdict_md, alert_text, json_text


def build_ui() -> gr.Blocks:
    """Gradio UI layout"""
    with gr.Blocks(title="Construction Safety Monitor", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
# 🦺 Construction Safety Monitor
Upload a construction site image to automatically detect PPE violations.
The model checks for **hard hats** and **high-visibility vests**.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image  = gr.Image(type="pil", label="Input image")
                conf_slider  = gr.Slider(0.1, 0.9, value=0.40, step=0.05,
                                         label="Detection confidence threshold")
                submit_btn   = gr.Button("Inspect scene", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(type="pil",  label="Annotated output")
                verdict_md   = gr.Markdown("### Upload an image to start")
                alert_box    = gr.Textbox(label="Violation report", lines=5,
                                          interactive=False)

        with gr.Accordion("JSON report (for developers)", open=False):
            json_output = gr.Code(language="json", label="Raw report")

        submit_btn.click(
            fn      = lambda img, c: predict(img),
            inputs  = [input_image, conf_slider],
            outputs = [output_image, verdict_md, alert_box, json_output],
        )

        gr.Examples(
            examples  = ['data/test_data/image_1.jpg', 
                         'data/test_data/image_2.jpg',
                         'data/test_data/image_5.jpg'],  
            inputs    = [input_image],
            label     = "Try an example",
        )

    return demo

def parse_args():
    p = argparse.ArgumentParser(description="Demo for construction safety")
    p.add_argument("--weights", default="data/model.pt", help="Path to model.pt")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--device", default=None,  help="'cuda', 'cpu', or device index")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public share link")
    return p.parse_args()

def main():
    args = parse_args()
    load_inspector(args.weights, args.conf, args.device)
    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
