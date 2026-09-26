"""
Brain Tumor Analysis — Gradio App

Two tabs:
  - Classification: EfficientNetB0 keras model (tumor / no tumor)
  - Detection: YOLOv8n model (bounding boxes around detected tumors)

Run from inside the llabelImg/ folder (see README_gradio.md for layout):
    python app_gradio.py

Research and educational use only. Not a medical device, not validated for
clinical use, and must not inform medical decisions.
"""

import os

import numpy as np
from PIL import Image
import gradio as gr

# --------------------------------------------------------------------------
# Configuration — edit these if your trained model files live elsewhere.
# --------------------------------------------------------------------------
CLASSIFICATION_MODEL_PATH = "brain_tumor_classification/buildEfficientB0_Model/trained_model_finetuned.keras"
DETECTION_MODEL_PATH = "brain_tumor_detection/runs/detect/train/weights/best.pt"

# Order assumed from the training generator (Keras image_dataset_from_directory
# sorts class subfolders alphabetically). Swap if predictions look inverted.
CLASS_NAMES = ["negative", "positive"]

# Matches the `240` argument passed to generate_augmented_dataset in train.ipynb.
CLASSIFICATION_IMG_SIZE = (240, 240)

# Matches the notebook's inference calls (model.predict(..., conf=0.25, ...)).
DEFAULT_DETECTION_CONF = 0.25


# --------------------------------------------------------------------------
# Lazy-loaded models — each is only loaded the first time its tab is used,
# so the app still starts (and the other tab still works) if one model file
# is missing.
# --------------------------------------------------------------------------
_classification_model = None
_detection_model = None


def _require_file(path: str, what: str):
    if not os.path.exists(path):
        raise gr.Error(
            f"Could not find the {what} model file at:\n  {os.path.abspath(path)}\n\n"
            f"Check CLASSIFICATION_MODEL_PATH / DETECTION_MODEL_PATH at the top of "
            f"app_gradio.py, or make sure you're running this script from the "
            f"llabelImg/ folder as described in README_gradio.md."
        )


def _get_classification_model():
    global _classification_model
    if _classification_model is None:
        _require_file(CLASSIFICATION_MODEL_PATH, "classification")
        import tensorflow as tf

        # Compatibility shim: the .keras file was saved with a Keras version
        # whose Dense layer config includes a `quantization_config` key. Some
        # installed Keras versions don't accept that constructor argument and
        # raise "Unrecognized keyword arguments passed to Dense" on load.
        # Passing a substitute class via custom_objects does NOT work here:
        # Keras resolves builtin classes like Dense by their registered
        # module path before ever consulting custom_objects, so the
        # substitute is silently ignored. Instead, patch the real Dense
        # class's __init__ for the duration of the load so it just absorbs
        # and ignores the extra keyword (we aren't using quantization, so
        # this is safe) — then restore the original afterward.
        _original_dense_init = tf.keras.layers.Dense.__init__

        def _patched_dense_init(self, *args, quantization_config=None, **kwargs):
            _original_dense_init(self, *args, **kwargs)

        tf.keras.layers.Dense.__init__ = _patched_dense_init
        try:
            _classification_model = tf.keras.models.load_model(
                CLASSIFICATION_MODEL_PATH, compile=False
            )
        finally:
            tf.keras.layers.Dense.__init__ = _original_dense_init
    return _classification_model


def _get_detection_model():
    global _detection_model
    if _detection_model is None:
        _require_file(DETECTION_MODEL_PATH, "detection")
        from ultralytics import YOLO

        _detection_model = YOLO(DETECTION_MODEL_PATH)
    return _detection_model


# ---------------------
# Classification tab
# ----------------------
def classify(image: Image.Image):
    if image is None:
        raise gr.Error("Please upload a brain MRI slice first.")

    model = _get_classification_model()

    img = image.convert("RGB").resize(CLASSIFICATION_IMG_SIZE)
    # Pixel values are left unscaled (0-255) — the model's own preprocess_input
    # layer (tf.keras.applications.efficientnet.preprocess_input) handles scaling.
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    verdict = f"{CLASS_NAMES[top_idx]} ({preds[top_idx] * 100:.1f}%)"
    probabilities = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    return verdict, probabilities


# ---------------------------
# Detection tab
# ---------------------------
def detect(image: Image.Image, conf: float):
    if image is None:
        raise gr.Error("Please upload a brain MRI slice first.")

    model = _get_detection_model()

    results = model.predict(np.array(image.convert("RGB")), conf=conf, verbose=False)
    result = results[0]

    annotated = Image.fromarray(result.plot()[:, :, ::-1])  # BGR -> RGB

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        summary = "No tumor regions detected above this confidence threshold."
    else:
        lines = []
        for i in range(len(boxes)):
            score = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = model.names.get(cls_id, str(cls_id))
            lines.append(f"Region {i + 1}: {cls_name} — {score * 100:.1f}% confidence")
        summary = "\n".join(lines)

    return annotated, summary


# ----------------------------
# UI
# ----------------------------
with gr.Blocks(title="Brain Tumor Analysis") as demo:
    gr.Markdown(
        "# Brain Tumor Analysis\n"
        "Research and educational use only. Not a medical device, not validated "
        "for clinical use, and must not inform medical decisions."
    )

    with gr.Tabs():
        with gr.Tab("Classification"):
            gr.Markdown("EfficientNetB0 — tumor / no tumor with per-class probabilities.")
            with gr.Row():
                with gr.Column():
                    cls_input = gr.Image(type="pil", label="MRI slice")
                    cls_button = gr.Button("Classify", variant="primary")
                with gr.Column():
                    cls_verdict = gr.Textbox(label="Verdict")
                    cls_probs = gr.Label(label="Class probabilities")

            cls_button.click(
                fn=classify, inputs=cls_input, outputs=[cls_verdict, cls_probs]
            )

        with gr.Tab("Detection"):
            gr.Markdown("YOLOv8n — draws bounding boxes around detected tumors.")
            with gr.Row():
                with gr.Column():
                    det_input = gr.Image(type="pil", label="MRI slice")
                    det_conf = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=DEFAULT_DETECTION_CONF,
                        step=0.05,
                        label="Detection confidence threshold",
                    )
                    det_button = gr.Button("Detect", variant="primary")
                with gr.Column():
                    det_output_image = gr.Image(label="Detected regions")
                    det_summary = gr.Textbox(label="Regions found", lines=4)

            det_button.click(
                fn=detect,
                inputs=[det_input, det_conf],
                outputs=[det_output_image, det_summary],
            )

if __name__ == "__main__":
    demo.launch()