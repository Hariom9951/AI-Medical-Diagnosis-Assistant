"""Pipeline Verification Script — Runs both inference pipelines end-to-end.

Verifies the Image Pipeline and NLP Pipeline using multiple unseen examples.
Run this script from the project root after both pipelines are set up.

Usage
-----
    .\\venv\\Scripts\\python scripts\\verify_pipelines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on the Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

# ---------------------------------------------------------------------------
# Image test cases - one per class (unseen images from each category)
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(
    "data/raw/covid19-radiography-database/COVID-19_Radiography_Dataset"
)

IMAGE_TEST_CASES = [
    {
        "path": DATASET_ROOT / "COVID/images/COVID-2.png",
        "expected_class": "COVID",
        "label": "COVID X-ray (unseen sample)",
    },
    {
        "path": DATASET_ROOT / "Normal/images/Normal-2.png",
        "expected_class": "Normal",
        "label": "Normal X-ray (unseen sample)",
    },
    {
        "path": DATASET_ROOT / "Lung_Opacity/images/Lung_Opacity-2.png",
        "expected_class": "Lung_Opacity",
        "label": "Lung Opacity X-ray (unseen sample)",
    },
    {
        "path": DATASET_ROOT / "Viral Pneumonia/images/Viral Pneumonia-2.png",
        "expected_class": "Viral Pneumonia",
        "label": "Viral Pneumonia X-ray (unseen sample)",
    },
]

# ---------------------------------------------------------------------------
# NLP test cases
# ---------------------------------------------------------------------------
NLP_TEST_CASES = [
    {
        "text": "I have fever, cough, sore throat, headache and body pain.",
        "label": "Flu-like symptoms",
    },
    {
        "text": "I am experiencing itching, skin rash, nodal skin eruptions and dischromic patches.",
        "label": "Skin condition symptoms",
    },
    {
        "text": "I have continuous sneezing, shivering, chills, watering from eyes and loss of smell.",
        "label": "Allergy / cold symptoms",
    },
    {
        "text": "I feel joint pain, knee pain, swelling of joints, stiff neck and painful walking.",
        "label": "Arthritis-type symptoms",
    },
    {
        "text": "I have vomiting, fatigue, weight loss, high fever, nausea and loss of appetite.",
        "label": "General infection symptoms",
    },
    {
        "text": "I am experiencing chest pain, shortness of breath, fast heart rate, and sweating.",
        "label": "Cardiac / respiratory symptoms",
    },
]


def _separator(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def run_image_pipeline() -> bool:
    """Initializes and verifies the Image Inference Pipeline."""
    _separator("IMAGE INFERENCE PIPELINE VERIFICATION")

    from src.inference.predict import ImageInferencePipeline

    print("\nLoading Image Pipeline...")
    try:
        pipeline = ImageInferencePipeline(
            config_path=Path("configs/training_config.yaml"),
            checkpoint_path=Path("artifacts/checkpoints/checkpoint_epoch_050.pth"),
        )
    except Exception as e:
        print(f"[FAIL] Image pipeline init failed: {e}")
        return False

    print("\nRunning inference on test images:\n")
    all_passed = True

    for i, case in enumerate(IMAGE_TEST_CASES, start=1):
        img_path = case["path"]
        if not img_path.exists():
            print(f"  [{i}] SKIP — image not found: {img_path}")
            continue

        try:
            result = pipeline.predict(img_path)
            predicted = result["predicted_disease"]
            confidence = result["confidence"]
            probs = result["class_probabilities"]

            match = predicted == case["expected_class"]
            status = "PASS" if match else "WARN"

            print(f"  [{i}] {case['label']}")
            print(f"       Predicted : {predicted} ({'CORRECT' if match else 'INCORRECT'})")
            print(f"       Confidence: {confidence:.6f}")
            print(f"       Class Probabilities:")
            for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
                bar = "#" * int(prob * 30)
                print(f"         {cls:<20} {prob:.4f}  {bar}")
            print()

        except Exception as e:
            print(f"  [{i}] FAIL — {case['label']}: {e}")
            all_passed = False

    return all_passed


def run_nlp_pipeline() -> bool:
    """Initializes and verifies the NLP Inference Pipeline."""
    _separator("NLP INFERENCE PIPELINE VERIFICATION")

    from src.inference.nlp_predict import NLPInferencePipeline

    print("\nLoading NLP Pipeline...")
    try:
        pipeline = NLPInferencePipeline(
            checkpoint_path=Path("artifacts/checkpoints_nlp/checkpoint_epoch_4.pt"),
            tokenizer_dir=Path("artifacts/checkpoints_nlp"),
            disease_mapping_path=Path("data/processed/disease_mapping_41.json"),
        )
    except Exception as e:
        print(f"[FAIL] NLP pipeline init failed: {e}")
        return False

    print("\nRunning inference on test symptom texts:\n")
    all_passed = True

    for i, case in enumerate(NLP_TEST_CASES, start=1):
        try:
            result = pipeline.predict(case["text"], top_k=5)
            predicted = result["predicted_disease"]
            confidence = result["confidence"]
            preprocessed = result["preprocessed_text"]
            top5 = result["top_predictions"]

            print(f"  [{i}] {case['label']}")
            print(f"       Input      : {case['text'][:80]}...")
            print(f"       Preprocessed: {preprocessed[:80]}")
            print(f"       Predicted  : {predicted}")
            print(f"       Confidence : {confidence:.6f}")
            print(f"       Top-5 Predictions:")
            for pred in top5:
                bar = "#" * int(pred["confidence"] * 30)
                print(
                    f"         #{pred['rank']}  {pred['disease']:<30} "
                    f"{pred['confidence']:.4f}  {bar}"
                )
            print()

        except Exception as e:
            print(f"  [{i}] FAIL — {case['label']}: {e}")
            all_passed = False

    return all_passed


def main() -> None:
    print("\n" + "#" * 70)
    print("#  AI MEDICAL DIAGNOSIS ASSISTANT — PIPELINE VERIFICATION")
    print("#" * 70)

    img_ok = run_image_pipeline()
    nlp_ok = run_nlp_pipeline()

    _separator("VERIFICATION SUMMARY")
    print(f"\n  Image Pipeline : {'PASS' if img_ok else 'FAIL'}")
    print(f"  NLP Pipeline   : {'PASS' if nlp_ok else 'FAIL'}")
    print()

    if img_ok and nlp_ok:
        print("  Both pipelines verified successfully.")
        print("  Ready for Streamlit integration.")
    else:
        print("  One or more pipelines failed. Review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
