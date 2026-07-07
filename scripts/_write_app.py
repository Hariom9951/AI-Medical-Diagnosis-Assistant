"""Writes the Streamlit frontend application."""

import pathlib

APP = r'''"""AI Medical Diagnosis Assistant — Streamlit Frontend.

Integrates two trained inference pipelines:
  - Mode 1: Chest X-ray Diagnosis  (EfficientNet-B0, 4 classes)
  - Mode 2: Symptom Diagnosis       (DistilBERT, 41 classes)

Run from project root:
    .\\venv\\Scripts\\streamlit run src/frontend/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so src.* imports work
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AI Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
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
</style>
""", unsafe_allow_html=True)


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
        checkpoint_path=Path("artifacts/checkpoints_nlp/checkpoint_epoch_4.pt"),
        tokenizer_dir=Path("artifacts/checkpoints_nlp"),
        disease_mapping_path=Path("data/processed/disease_mapping_41.json"),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def confidence_color(conf: float) -> str:
    """Returns a CSS colour string based on confidence level."""
    if conf >= 0.75:
        return "#34d399"   # green
    elif conf >= 0.45:
        return "#fbbf24"   # amber
    else:
        return "#f87171"   # red


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
        st.progress(float(confidence))
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
        '<p style="color:rgba(255,255,255,0.5);font-size:0.85rem;margin:0">Assistant v1.0</p>'
        '</div>',
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

    st.markdown("""
    <div style="color:rgba(255,255,255,0.5);font-size:0.8rem">
    <b style="color:rgba(255,255,255,0.7)">Models</b><br>
    🖼 EfficientNet-B0 (4 classes)<br>
    📝 DistilBERT (41 diseases)<br><br>
    <b style="color:rgba(255,255,255,0.7)">Checkpoints</b><br>
    Epoch 50 · Val Acc 100%<br>
    Epoch 4 · Val Acc 100%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="color:rgba(255,255,255,0.3);font-size:0.75rem;text-align:center">'
        'For research use only</p>',
        unsafe_allow_html=True,
    )


# ── Hero Banner ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <h1>🏥 AI Medical Diagnosis Assistant</h1>
    <p>Powered by EfficientNet-B0 & DistilBERT — Advanced deep learning for medical screening</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE 1 — CHEST X-RAY DIAGNOSIS
# ════════════════════════════════════════════════════════════════════════════

if "Chest X-ray" in mode:

    st.markdown(
        '<p class="section-header">🫁 Chest X-ray Diagnosis</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="info-box">
    Upload a chest X-ray image (PNG, JPG, JPEG). The model will predict whether
    the scan shows <b>COVID-19</b>, <b>Lung Opacity</b>, <b>Normal</b>, or
    <b>Viral Pneumonia</b> using a trained EfficientNet-B0 classifier.
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG",
            label_visibility="collapsed",
        )

        run_image = st.button("🔍  Analyze X-ray", use_container_width=True)

    with col_preview:
        if uploaded_file is not None:
            st.image(
                uploaded_file,
                caption="Uploaded X-ray",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div style="border:2px dashed rgba(255,255,255,0.15);border-radius:12px;
                        padding:3rem;text-align:center;color:rgba(255,255,255,0.3)">
                <div style="font-size:2.5rem;margin-bottom:0.5rem">🩻</div>
                <div>X-ray preview will appear here</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Run inference ────────────────────────────────────────────────────────
    if run_image:
        if uploaded_file is None:
            st.markdown("""
            <div class="result-card-error">
            ⚠️ <b>No image uploaded.</b> Please upload a chest X-ray image before running analysis.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Loading image model & running inference..."):
                try:
                    pipeline = load_image_pipeline()
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(uploaded_file)
                    result = pipeline.predict(pil_img)

                    predicted = result["predicted_disease"]
                    confidence = result["confidence"]
                    class_probs = result["class_probabilities"]

                    # Sort all classes by probability descending; show top 3
                    sorted_preds = sorted(
                        class_probs.items(), key=lambda x: x[1], reverse=True
                    )[:3]

                except Exception as e:
                    st.markdown(
                        f'<div class="result-card-error">❌ <b>Inference failed:</b> {e}</div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()

            # ── Results display ─────────────────────────────────────────────
            st.markdown('<p class="section-header">📊 Diagnosis Results</p>', unsafe_allow_html=True)

            # Top metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Predicted Condition</div>
                    <div class="disease-name">{predicted}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                conf_pct = f"{confidence*100:.2f}%"
                col = confidence_color(confidence)
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value" style="color:{col}">{conf_pct}</div>
                    <div class="metric-sub">Model certainty</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                classes_str = " · ".join([p[0] for p in sorted_preds[:3]])
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Top Conditions</div>
                    <div style="color:rgba(255,255,255,0.8);font-size:0.85rem;
                                font-weight:500;margin-top:0.4rem">{classes_str}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Top-3 prediction bars
            st.markdown(
                '<p style="color:rgba(255,255,255,0.7);font-weight:600;'
                'margin-bottom:0.8rem">Top-3 Class Probabilities</p>',
                unsafe_allow_html=True,
            )
            for rank, (disease, prob) in enumerate(sorted_preds, start=1):
                render_prediction_bar(rank, disease, prob)

            # Confidence level badge
            if confidence >= 0.75:
                level, badge_col = "High Confidence", "#34d399"
            elif confidence >= 0.45:
                level, badge_col = "Moderate Confidence", "#fbbf24"
            else:
                level, badge_col = "Low Confidence", "#f87171"

            st.markdown(
                f'<div style="margin-top:1rem;padding:0.6rem 1rem;background:rgba(255,255,255,0.05);'
                f'border-radius:8px;color:{badge_col};font-weight:600;font-size:0.9rem">'
                f'⚡ {level} — {confidence*100:.1f}%</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# MODE 2 — SYMPTOM DIAGNOSIS
# ════════════════════════════════════════════════════════════════════════════

else:

    st.markdown(
        '<p class="section-header">💊 Symptom-Based Diagnosis</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="info-box">
    Describe your symptoms in plain English. The model will analyse the text
    using a fine-tuned <b>DistilBERT</b> transformer and predict the most likely
    condition from <b>41 diseases</b>.
    </div>
    """, unsafe_allow_html=True)

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
        'Quick examples:</p>',
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(len(EXAMPLES))
    chosen_example = None
    for i, (col, ex) in enumerate(zip(ex_cols, EXAMPLES)):
        with col:
            if st.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
                chosen_example = ex

    symptom_input = st.text_area(
        "Describe your symptoms",
        value=chosen_example if chosen_example else "",
        height=130,
        placeholder="e.g. I have fever, cough, sore throat, headache and body pain.",
        label_visibility="collapsed",
    )

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        run_nlp = st.button("🔍  Diagnose Symptoms", use_container_width=True)
    with col_clear:
        if st.button("✕  Clear", use_container_width=True):
            symptom_input = ""

    # ── Run inference ────────────────────────────────────────────────────────
    if run_nlp:
        if not symptom_input or not symptom_input.strip():
            st.markdown("""
            <div class="result-card-error">
            ⚠️ <b>No symptoms entered.</b> Please describe your symptoms before running analysis.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Loading NLP model & running inference..."):
                try:
                    pipeline = load_nlp_pipeline()
                    result = pipeline.predict(symptom_input.strip(), top_k=5)

                    predicted = result["predicted_disease"]
                    confidence = result["confidence"]
                    top5 = result["top_predictions"]
                    preprocessed = result["preprocessed_text"]

                except Exception as e:
                    st.markdown(
                        f'<div class="result-card-error">❌ <b>Inference failed:</b> {e}</div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()

            # ── Results display ─────────────────────────────────────────────
            st.markdown('<p class="section-header">📊 Diagnosis Results</p>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Predicted Condition</div>
                    <div class="disease-name">{predicted}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                conf_pct = f"{confidence*100:.2f}%"
                col = confidence_color(confidence)
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value" style="color:{col}">{conf_pct}</div>
                    <div class="metric-sub">Model certainty</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                num_words = len(symptom_input.split())
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Input Analysed</div>
                    <div class="metric-value" style="font-size:1.4rem">{num_words}</div>
                    <div class="metric-sub">words processed</div>
                </div>
                """, unsafe_allow_html=True)

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

            # Confidence badge
            if confidence >= 0.75:
                level, badge_col = "High Confidence", "#34d399"
            elif confidence >= 0.45:
                level, badge_col = "Moderate Confidence", "#fbbf24"
            else:
                level, badge_col = "Low Confidence — Consider alternative diagnoses", "#f87171"

            st.markdown(
                f'<div style="margin-top:1rem;padding:0.6rem 1rem;background:rgba(255,255,255,0.05);'
                f'border-radius:8px;color:{badge_col};font-weight:600;font-size:0.9rem">'
                f'⚡ {level} — {confidence*100:.1f}%</div>',
                unsafe_allow_html=True,
            )


# ── Medical disclaimer ───────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚕️ <b>Medical Disclaimer:</b> This tool is for research and educational purposes only.
It is <b>not</b> a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
'''

pathlib.Path("src/frontend/app.py").write_text(APP, encoding="utf-8")
print("src/frontend/app.py written OK")
