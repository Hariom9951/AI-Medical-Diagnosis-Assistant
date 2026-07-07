"""AI Medical Diagnosis Assistant — Streamlit Frontend.

Integrates two trained inference pipelines:
  - Mode 1: Chest X-ray Diagnosis  (EfficientNet-B0, 4 classes)
  - Mode 2: Symptom Diagnosis       (DistilBERT, 41 classes)

And incorporates:
  - Professional PDF Medical Report Generator (ReportLab)
  - Grad-CAM Explainability for Chest X-ray Model (PyTorch Grad-CAM)

Run from project root:
    .\\venv\\Scripts\\streamlit run src/frontend/app.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so src.* imports work
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.report.pdf_generator import MedicalReportGenerator

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AI Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
}
.hero-banner h1 {
    color: white;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255,255,255,0.85);
    font-size: 1.1rem;
    margin: 0;
}

/* Mode card */
.mode-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.mode-card:hover {
    border-color: rgba(102, 126, 234, 0.6);
    box-shadow: 0 4px 24px rgba(102, 126, 234, 0.2);
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 14px;
    padding: 1.8rem;
    margin: 1.5rem 0;
}
.result-card-error {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

/* Metric tiles */
.metric-tile {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.metric-value {
    color: #a78bfa;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-sub {
    color: rgba(255,255,255,0.5);
    font-size: 0.75rem;
    margin-top: 0.3rem;
}

/* Disease name display */
.disease-name {
    color: #34d399;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.2rem 0 0.5rem 0;
}

/* Prediction bar row */
.pred-bar-container {
    margin: 0.6rem 0;
}
.pred-rank-badge {
    display: inline-block;
    background: rgba(102,126,234,0.3);
    color: #a5b4fc;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    text-align: center;
    line-height: 24px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
}
.pred-disease-label {
    color: rgba(255,255,255,0.9);
    font-weight: 500;
    font-size: 0.95rem;
}
.pred-pct {
    color: #a78bfa;
    font-weight: 600;
    font-size: 0.9rem;
    float: right;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: opacity 0.2s ease, transform 0.1s ease;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* Info box */
.info-box {
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: rgba(255,255,255,0.8);
    font-size: 0.9rem;
    margin: 1rem 0;
}

/* Warning disclaimer */
.disclaimer {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: rgba(255,255,255,0.7);
    font-size: 0.82rem;
    margin-top: 2rem;
    text-align: center;
}

/* Section header */
.section-header {
    color: white;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 1.5rem 0 0.8rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 0.5rem;
}

/* Progress bar override */
.stProgress > div > div {
    border-radius: 8px;
}

/* High Contrast Medical Cards */
.medical-card {
    background: rgba(30, 41, 59, 0.75) !important;
    border: 1px solid rgba(147, 197, 253, 0.3) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    color: #f8fafc !important; /* high contrast white/light-grey */
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    margin-bottom: 1.5rem !important;
}

.warning-card {
    background: rgba(127, 29, 29, 0.45) !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    color: #fee2e2 !important; /* light red/white text */
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 1.5rem !important;
}

/* Bullet list custom styling for high contrast */
.medical-card ul, .warning-card ul {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    padding-left: 1.2rem !important;
}
.medical-card li, .warning-card li {
    color: #f8fafc !important;
    margin-bottom: 0.3rem !important;
}

/* PDF download button styling override */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important; /* Emerald green for download/reports */
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
    opacity: 0.95 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

import json
from typing import Any, Dict


def get_model_metadata() -> Dict[str, Any]:
    """Dynamically reads and evaluates NLP and Image model checkpoint metadata."""
    metadata = {
        "image_model": "EfficientNet-B0",
        "image_classes": "4 classes",
        "image_epoch": "N/A",
        "image_val_acc": "N/A",
        "nlp_model": "BioBERT Medical Classifier",
        "nlp_classes": "41 diseases",
        "nlp_checkpoint": "best_model.pt",
        "nlp_epoch": "N/A",
        "nlp_val_acc": "N/A",
    }

    # 1. Load Image Checkpoint metadata
    img_ckpt = Path("artifacts/checkpoints/checkpoint_epoch_050.pth")
    if img_ckpt.exists():
        try:
            import torch

            checkpoint_data = torch.load(img_ckpt, map_location="cpu", weights_only=False)
            if isinstance(checkpoint_data, dict):
                ep = checkpoint_data.get("epoch", None)
                if ep is not None:
                    metadata["image_epoch"] = str(int(ep) + 1)
                metrics = checkpoint_data.get("metrics", {})
                val_acc = metrics.get("val_acc", None)
                if val_acc is not None:
                    metadata["image_val_acc"] = f"{float(val_acc):.2f}%"
        except Exception:
            pass

    # 2. Load NLP Checkpoint metadata from model_metadata.json
    nlp_meta_path = Path("artifacts/checkpoints_nlp/model_metadata.json")
    if nlp_meta_path.exists():
        try:
            with open(nlp_meta_path, "r", encoding="utf-8") as f:
                nlp_meta = json.load(f)
            metadata["nlp_model"] = nlp_meta.get("model_name", metadata["nlp_model"])
            metadata["nlp_checkpoint"] = nlp_meta.get("checkpoint", metadata["nlp_checkpoint"])

            num_classes = nlp_meta.get("num_classes", None)
            if num_classes is not None:
                metadata["nlp_classes"] = f"{num_classes} diseases"

            val_acc = nlp_meta.get("validation_accuracy", None)
            if val_acc is not None:
                metadata["nlp_val_acc"] = f"{val_acc}%"

            epoch = nlp_meta.get("epoch", None)
            if epoch is not None:
                metadata["nlp_epoch"] = str(epoch)
        except Exception:
            pass

    return metadata


# ── Cached pipeline loaders ──────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_image_pipeline():
    """Loads and caches the Image Inference Pipeline."""
    from src.inference.predict import ImageInferencePipeline

    return ImageInferencePipeline(
        config_path=Path("configs/training_config.yaml"),
        checkpoint_path=Path("artifacts/checkpoints/checkpoint_epoch_050.pth"),
    )


@st.cache_resource(show_spinner=False)
def load_nlp_pipeline():
    """Loads and caches the NLP Inference Pipeline."""
    from src.inference.nlp_predict import NLPInferencePipeline

    return NLPInferencePipeline(
        checkpoint_path=Path("artifacts/checkpoints_nlp/best_model.pt"),
        tokenizer_dir=Path("artifacts/checkpoints_nlp"),
        disease_mapping_path=Path("data/processed/disease_mapping_41.json"),
    )


def handle_xray_upload_change():
    """Callback function to handle st.file_uploader upload/clear actions."""
    import logging

    logger = logging.getLogger("src.frontend.app")
    uploaded_file = st.session_state.get("xray_uploader")
    if uploaded_file is not None:
        logger.info(
            "xray_uploader callback: new file uploaded name=%s size=%d bytes",
            uploaded_file.name,
            uploaded_file.size,
        )
        st.session_state.current_image_bytes = uploaded_file.getvalue()
        st.session_state.current_image_name = uploaded_file.name
    else:
        logger.info("xray_uploader callback: file cleared by user.")
        st.session_state.current_image_bytes = None
        st.session_state.current_image_name = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def confidence_color(conf: float) -> str:
    """Returns a CSS colour string based on confidence level."""
    if conf >= 0.75:
        return "#34d399"  # green
    elif conf >= 0.45:
        return "#fbbf24"  # amber
    else:
        return "#f87171"  # red


def render_prediction_bar(rank: int, disease: str, confidence: float) -> None:
    """Renders a single ranked prediction row with a progress bar."""
    col_label, col_bar, col_pct = st.columns([3, 5, 1])
    with col_label:
        st.markdown(
            f'<span class="pred-rank-badge">{rank}</span>'
            f'<span class="pred-disease-label">{disease}</span>',
            unsafe_allow_html=True,
        )
    with col_bar:
        st.progress(confidence)
    with col_pct:
        st.markdown(
            f'<div style="color:{confidence_color(confidence)};font-weight:600;'
            f'font-size:0.9rem;padding-top:6px">{confidence*100:.1f}%</div>',
            unsafe_allow_html=True,
        )


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;margin-bottom:1.5rem">'
        '<span style="font-size:2.5rem">🏥</span>'
        '<h2 style="color:white;margin:0.3rem 0 0 0;font-size:1.3rem">AI Diagnosis</h2>'
        '<p style="color:rgba(255,255,255,0.5);font-size:0.85rem;margin:0">Assistant v2.0 (BioBERT)</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    mode = st.radio(
        "**Diagnosis Mode**",
        options=["🫁  Chest X-ray Diagnosis", "💊  Symptom Diagnosis"],
        index=0,
        help="Choose whether to diagnose from an X-ray image or text symptoms.",
    )

    st.markdown("---")

    meta = get_model_metadata()

    st.markdown(
        f"""
    <div style="color:rgba(255,255,255,0.65);font-size:0.8rem;line-height:1.4">
    <b style="color:white;font-size:0.85rem">🧠 MODEL METADATA</b><br><br>
    🖼️ <b>Image Model</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;{meta['image_model']} ({meta['image_classes']})<br>
    &nbsp;&nbsp;&nbsp;&nbsp;Checkpoint: checkpoint_epoch_050.pth<br>
    &nbsp;&nbsp;&nbsp;&nbsp;Epoch: {meta['image_epoch']}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;Val Accuracy: {meta['image_val_acc']}<br><br>
    📝 <b>NLP Model</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;{meta['nlp_model']} ({meta['nlp_classes']})<br>
    &nbsp;&nbsp;&nbsp;&nbsp;Checkpoint: {meta['nlp_checkpoint']}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;Epoch: {meta['nlp_epoch']}<br>
    &nbsp;&nbsp;&nbsp;&nbsp;Val Accuracy: {meta['nlp_val_acc']}<br>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<p style="color:rgba(255,255,255,0.3);font-size:0.75rem;text-align:center">'
        "For research use only</p>",
        unsafe_allow_html=True,
    )


# ── Hero Banner ──────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero-banner">
    <h1>🏥 AI Medical Diagnosis Assistant</h1>
    <p>Powered by EfficientNet-B0 & BioBERT — Advanced deep learning for medical screening</p>
</div>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

if "image_results" not in st.session_state:
    st.session_state.image_results = None
if "image_inference_time" not in st.session_state:
    st.session_state.image_inference_time = None
if "image_report_path" not in st.session_state:
    st.session_state.image_report_path = None
if "last_image_name" not in st.session_state:
    st.session_state.last_image_name = None

# Grad-CAM specific states
if "image_heatmap_img" not in st.session_state:
    st.session_state.image_heatmap_img = None
if "image_overlay_img" not in st.session_state:
    st.session_state.image_overlay_img = None
if "image_overlay_path" not in st.session_state:
    st.session_state.image_overlay_path = None

if "nlp_results" not in st.session_state:
    st.session_state.nlp_results = None
if "nlp_inference_time" not in st.session_state:
    st.session_state.nlp_inference_time = None
if "nlp_report_path" not in st.session_state:
    st.session_state.nlp_report_path = None
if "last_nlp_input" not in st.session_state:
    st.session_state.last_nlp_input = None


# ════════════════════════════════════════════════════════════════════════════
# MODE 1 — CHEST X-RAY DIAGNOSIS
# ════════════════════════════════════════════════════════════════════════════

if "Chest X-ray" in mode:

    st.markdown(
        '<p class="section-header">🫁 Chest X-ray Diagnosis</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
    Upload a chest X-ray image (PNG, JPG, JPEG). The model will predict whether
    the scan shows <b>COVID-19</b>, <b>Lung Opacity</b>, <b>Normal</b>, or
    <b>Viral Pneumonia</b> using a trained EfficientNet-B0 classifier.
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_upload, col_preview = st.columns([1, 1])

    import logging

    app_logger = logging.getLogger("src.frontend.app")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG",
            label_visibility="collapsed",
            key="xray_uploader",
            on_change=handle_xray_upload_change,
        )

        # Log active cache state
        if st.session_state.get("current_image_bytes") is not None:
            app_logger.info(
                "Active image file in session cache: name=%s, size=%d bytes",
                st.session_state.current_image_name,
                len(st.session_state.current_image_bytes),
            )
        else:
            app_logger.debug("No active image in session cache.")

        run_image = st.button("🔍  Analyze X-ray", use_container_width=True)

    with col_preview:
        if st.session_state.get("current_image_bytes") is not None:
            # If the user uploaded a different image, reset previous results
            if st.session_state.last_image_name != st.session_state.current_image_name:
                st.session_state.image_results = None
                st.session_state.image_inference_time = None
                st.session_state.image_report_path = None
                st.session_state.image_heatmap_img = None
                st.session_state.image_overlay_img = None
                st.session_state.image_overlay_path = None
                st.session_state.last_image_name = st.session_state.current_image_name

            st.image(
                st.session_state.current_image_bytes,
                caption="Uploaded X-ray",
                use_container_width=True,
            )
        else:
            st.session_state.image_results = None
            st.session_state.image_inference_time = None
            st.session_state.image_report_path = None
            st.session_state.image_heatmap_img = None
            st.session_state.image_overlay_img = None
            st.session_state.image_overlay_path = None
            st.session_state.last_image_name = None

            st.markdown(
                """
            <div style="border:2px dashed rgba(255,255,255,0.15);border-radius:12px;
                        padding:3rem;text-align:center;color:rgba(255,255,255,0.3)">
                <div style="font-size:2.5rem;margin-bottom:0.5rem">🩻</div>
                <div>X-ray preview will appear here</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ── Run inference ────────────────────────────────────────────────────────
    if run_image:
        if st.session_state.get("current_image_bytes") is None:
            st.markdown(
                """
            <div class="result-card-error">
            ⚠️ <b>No image uploaded.</b> Please upload a chest X-ray image before running analysis.
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Loading image model & running inference..."):
                try:
                    app_logger.info("Initializing image inference pipeline...")
                    pipeline = load_image_pipeline()
                    import io

                    from PIL import Image as PILImage

                    app_logger.info("Preprocessing uploaded image for model prediction...")
                    pil_img = PILImage.open(
                        io.BytesIO(st.session_state.current_image_bytes)
                    ).convert("RGB")

                    # Predict
                    start_time = time.perf_counter()
                    app_logger.info("Executing prediction on the preprocessed image...")
                    result = pipeline.predict(pil_img)
                    end_time = time.perf_counter()

                    app_logger.info(
                        "Inference complete: Predicted=%s, Confidence=%.4f",
                        result["predicted_disease"],
                        result["confidence"],
                    )
                    st.session_state.image_inference_time = (end_time - start_time) * 1000
                    st.session_state.image_results = result

                    # 1. Generate Grad-CAM heatmap visualization
                    try:
                        from src.explainability.gradcam import GradCAMExplainer
                        from src.explainability.visualizer import GradCAMVisualizer

                        # Build model target index
                        class_probs = result["class_probabilities"]
                        sorted_preds = sorted(
                            class_probs.items(), key=lambda x: x[1], reverse=True
                        )[:3]

                        # Map prediction label to its class index in model's order
                        # Model Classes: ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
                        pred_disease = result["predicted_disease"]
                        pred_idx = pipeline.CLASSES.index(pred_disease)

                        # Generate heatmap using preprocessed tensor
                        img_tensor = pipeline.preprocess(pil_img)
                        explainer = GradCAMExplainer(model=pipeline.model)
                        heatmap = explainer.generate_heatmap(img_tensor, target_class_idx=pred_idx)

                        if heatmap is not None:
                            visualizer = GradCAMVisualizer()
                            heatmap_img, overlay_img, overlay_path = (
                                visualizer.create_visualization(
                                    original_image=pil_img, heatmap=heatmap, alpha=0.6
                                )
                            )
                            st.session_state.image_heatmap_img = heatmap_img
                            st.session_state.image_overlay_img = overlay_img
                            st.session_state.image_overlay_path = str(overlay_path)
                    except Exception as gcam_err:
                        st.warning(f"Grad-CAM explanation generation failed: {gcam_err}")

                    # 2. Generate professional PDF report
                    report_gen = MedicalReportGenerator()
                    predictions_list = [
                        {"rank": i + 1, "disease": k, "confidence": v}
                        for i, (k, v) in enumerate(sorted_preds)
                    ]

                    pdf_path = report_gen.generate_report(
                        mode="Chest X-ray Diagnosis",
                        user_input=f"Uploaded Image: {uploaded_file.name}",
                        predicted_disease=result["predicted_disease"],
                        confidence=result["confidence"],
                        predictions=predictions_list,
                        model_used="EfficientNet-B0",
                        inference_time_ms=st.session_state.image_inference_time,
                    )
                    st.session_state.image_report_path = str(pdf_path)

                except Exception as e:
                    st.markdown(
                        f'<div class="result-card-error">❌ <b>Inference failed:</b> {e}</div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()

    # ── Results display ─────────────────────────────────────────────
    if st.session_state.image_results is not None:
        result = st.session_state.image_results
        predicted = result["predicted_disease"]
        confidence = result["confidence"]
        class_probs = result["class_probabilities"]
        inference_time_ms = st.session_state.image_inference_time
        pdf_path_str = st.session_state.image_report_path

        sorted_preds = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:3]

        st.markdown('<p class="section-header">📊 Diagnosis Results</p>', unsafe_allow_html=True)

        # Top metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""
            <div class="metric-tile">
                <div class="metric-label">Predicted Condition</div>
                <div class="disease-name">{predicted}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m2:
            conf_pct = f"{confidence*100:.2f}%"
            col = confidence_color(confidence)
            st.markdown(
                f"""
            <div class="metric-tile">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value" style="color:{col}">{conf_pct}</div>
                <div class="metric-sub">Model certainty</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
            <div class="metric-tile">
                <div class="metric-label">Inference Time</div>
                <div class="metric-value" style="font-size:1.6rem">{inference_time_ms:.1f} ms</div>
                <div class="metric-sub">EfficientNet-B0 execution</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Top-3 prediction bars
        st.markdown(
            '<p style="color:rgba(255,255,255,0.7);font-weight:600;'
            'margin-bottom:0.8rem">Top-3 Class Probabilities</p>',
            unsafe_allow_html=True,
        )
        for rank, (disease, prob) in enumerate(sorted_preds, start=1):
            render_prediction_bar(rank, disease, prob)

        # ── Grad-CAM AI Explanation Block ─────────────────────────────────────
        if st.session_state.image_heatmap_img is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            show_explanation = st.checkbox("🔍 Show AI Explanation (Grad-CAM)")

            if show_explanation:
                st.markdown(
                    '<p style="color:rgba(255,255,255,0.7);font-weight:600;'
                    'margin-bottom:0.8rem">Grad-CAM Visual Attention Mapping</p>',
                    unsafe_allow_html=True,
                )

                # Show 3 columns: Original, Heatmap, Overlay
                col_orig, col_heat, col_over = st.columns(3)
                with col_orig:
                    # Retrieve the original uploaded image file directly to display
                    from PIL import Image as PILImage

                    if uploaded_file is not None:
                        pil_img = PILImage.open(uploaded_file)
                        st.image(pil_img, caption="Original X-ray Scan", use_container_width=True)
                    else:
                        st.warning("Please upload the chest X-ray image to preview.")
                with col_heat:
                    st.image(
                        st.session_state.image_heatmap_img,
                        caption="Attention Heatmap (Conv2D Activation)",
                        use_container_width=True,
                    )
                with col_over:
                    st.image(
                        st.session_state.image_overlay_img,
                        caption=f"Attention Overlay on {predicted}",
                        use_container_width=True,
                    )

        # Download Report Block
        if pdf_path_str and Path(pdf_path_str).exists():
            st.markdown("<br>", unsafe_allow_html=True)
            with open(pdf_path_str, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download Professional PDF Medical Report",
                data=pdf_bytes,
                file_name=Path(pdf_path_str).name,
                mime="application/pdf",
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# MODE 2 — SYMPTOM DIAGNOSIS
# ════════════════════════════════════════════════════════════════════════════

else:

    st.markdown(
        '<p class="section-header">💊 Symptom-Based Diagnosis</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
    Describe your symptoms in plain English. The model will analyse the text
    using a fine-tuned <b>BioBERT</b> transformer and predict the most likely
    condition from <b>41 diseases</b>.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Example chips
    EXAMPLES = [
        "I have fever, cough, sore throat, headache and body pain.",
        "I am experiencing itching, skin rash, nodal skin eruptions and dischromic patches.",
        "I feel joint pain, knee pain, swelling of joints, stiff neck and painful walking.",
        "I have continuous sneezing, shivering, chills, watering from eyes and loss of smell.",
        "I am experiencing chest pain, shortness of breath, fast heart rate, and sweating.",
    ]

    st.markdown(
        '<p style="color:rgba(255,255,255,0.5);font-size:0.85rem;margin-bottom:0.4rem">'
        "Quick examples:</p>",
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(len(EXAMPLES))
    cleaned_input = ""
    symptom_input = ""

    # Display examples as clickable buttons
    for i, ex in enumerate(EXAMPLES):
        with ex_cols[i]:
            if st.button(ex[:20] + "...", key=f"ex_{i}", help=ex):
                st.session_state.symptom_text_val = ex

    # Managed input text area
    symptom_input = st.text_area(
        "**Describe Symptoms**",
        value=st.session_state.get("symptom_text_val", ""),
        height=140,
        placeholder="Type your symptoms here (e.g., 'cough, high fever, running nose, shivering...')",
    )

    cleaned_input = (symptom_input or "").strip()

    c1, c2 = st.columns([1, 4])
    with c1:
        run_nlp = st.button("🔍 Analyze Symptoms")
    with c2:
        if st.button("🧹 Clear Input"):
            st.session_state.symptom_text_val = ""
            st.session_state.nlp_results = None
            st.session_state.nlp_inference_time = None
            st.session_state.nlp_report_path = None
            st.rerun()

    # ── Run inference ────────────────────────────────────────────────────────
    if run_nlp:
        if not cleaned_input:
            st.markdown(
                """
            <div class="result-card-error">
            ⚠️ <b>No symptoms entered.</b> Please describe your symptoms before running analysis.
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Loading NLP model & running inference..."):
                try:
                    pipeline = load_nlp_pipeline()

                    start_time = time.perf_counter()
                    result = pipeline.predict(cleaned_input, top_k=5)
                    end_time = time.perf_counter()

                    st.session_state.nlp_inference_time = (end_time - start_time) * 1000
                    st.session_state.nlp_results = result

                    # Generate professional PDF report
                    report_gen = MedicalReportGenerator()
                    pdf_path = report_gen.generate_report(
                        mode="Symptom Diagnosis",
                        user_input=cleaned_input,
                        predicted_disease=result["predicted_disease"],
                        confidence=result["confidence"],
                        predictions=result["top_predictions"],
                        model_used="BioBERT",
                        inference_time_ms=st.session_state.nlp_inference_time,
                    )
                    st.session_state.nlp_report_path = str(pdf_path)

                except Exception as e:
                    st.markdown(
                        f'<div class="result-card-error">❌ <b>Inference failed:</b> {e}</div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()

    # ── Results display ─────────────────────────────────────────────
    if st.session_state.nlp_results is not None:
        result = st.session_state.nlp_results
        predicted = result["predicted_disease"]
        confidence = result["confidence"]
        top5 = result["top_predictions"]
        preprocessed = result["preprocessed_text"]
        inference_time_ms = st.session_state.nlp_inference_time
        pdf_path_str = st.session_state.nlp_report_path

        st.markdown('<p class="section-header">📊 Diagnosis Results</p>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""
            <div class="metric-tile">
                <div class="metric-label">Predicted Condition</div>
                <div class="disease-name">{predicted}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m2:
            conf_pct = f"{confidence*100:.2f}%"
            col = confidence_color(confidence)
            st.markdown(
                f"""
            <div class="metric-tile">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value" style="color:{col}">{conf_pct}</div>
                <div class="metric-sub">Model certainty</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
            <div class="metric-tile">
                <div class="metric-label">Inference Time</div>
                <div class="metric-value" style="font-size:1.6rem">{inference_time_ms:.1f} ms</div>
                <div class="metric-sub">BioBERT execution</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Preprocessed text expander
        with st.expander("🔤 View preprocessed input"):
            st.markdown(
                f'<code style="color:#a5b4fc;font-size:0.9rem">{preprocessed}</code>',
                unsafe_allow_html=True,
            )

        # Top-5 prediction bars
        st.markdown(
            '<p style="color:rgba(255,255,255,0.7);font-weight:600;'
            'margin-bottom:0.8rem">Top-5 Disease Predictions</p>',
            unsafe_allow_html=True,
        )
        for pred in top5:
            render_prediction_bar(pred["rank"], pred["disease"], pred["confidence"])

        # Render clinical explanations if available (BioBERT-specific features)
        explanation = result.get("clinical_explanation", None)
        if explanation:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<p class="section-header">🩺 Clinical Explanation & Guidance</p>',
                unsafe_allow_html=True,
            )

            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.markdown(
                    f"""
                <div class="medical-card" style="height: 100%;">
                    <h4 style="color:#60a5fa;margin-top:0;margin-bottom:0.8rem">📋 Medical Guidance</h4>
                    <p style="margin-bottom:0.5rem"><b>Recommended Specialist:</b> {explanation['specialist']}</p>
                    <p style="margin-bottom:0.3rem"><b>Suggested Diagnostic Tests:</b></p>
                    <ul style="margin-top:0; margin-bottom:0.5rem">
                        {''.join(f"<li>{test}</li>" for test in explanation['tests'])}
                    </ul>
                    <p style="margin-bottom:0"><b>Similar Conditions to Rule Out:</b> {', '.join(explanation['similar_diseases'])}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with e_col2:
                # Highlight warning signs if critical
                st.markdown(
                    f"""
                <div class="warning-card" style="height: 100%;">
                    <h4 style="color:#f87171;margin-top:0;margin-bottom:0.8rem">⚠️ Emergency Warning Signs</h4>
                    <p style="color:#fbcfe8; margin-bottom:0.8rem"><b>Warning Signs:</b> {explanation['emergency_signs']}</p>
                    <h4 style="color:#34d399;margin-top:0.8rem;margin-bottom:0.5rem">🏡 Home Care & Lifestyle</h4>
                    <p style="margin-bottom:0.4rem"><b>Care:</b> {explanation['home_care']}</p>
                    <p style="margin-bottom:0"><b>Lifestyle:</b> {explanation['lifestyle']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Download Report Block
        if pdf_path_str and Path(pdf_path_str).exists():
            st.markdown("<br>", unsafe_allow_html=True)
            with open(pdf_path_str, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download Professional PDF Medical Report",
                data=pdf_bytes,
                file_name=Path(pdf_path_str).name,
                mime="application/pdf",
                use_container_width=True,
            )


# ── Medical disclaimer ───────────────────────────────────────────────────────
st.markdown(
    """
<div class="disclaimer">
⚕️ <b>Medical Disclaimer:</b> This tool is for research and educational purposes only.
It is <b>not</b> a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider.
</div>
""",
    unsafe_allow_html=True,
)
