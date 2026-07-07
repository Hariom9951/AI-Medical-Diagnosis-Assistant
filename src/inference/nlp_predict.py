from __future__ import annotations

import json
import re
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    AutoTokenizer,
)

from src.utils.exceptions import AppInferenceError, AppValidationError
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants - must match training configuration exactly
# ---------------------------------------------------------------------------
_DEFAULT_CHECKPOINT = Path("artifacts/checkpoints_nlp/best_model.pt")
_DEFAULT_TOKENIZER_DIR = Path("artifacts/checkpoints_nlp")
_DEFAULT_DISEASE_MAPPING = Path("data/processed/disease_mapping_41.json")
_DEFAULT_MAX_LENGTH: int = 128
_DEFAULT_MODEL_NAME: str = "dmis-lab/biobert-base-cased-v1.1"
_DEFAULT_DROPOUT: float = 0.2


# ---------------------------------------------------------------------------
# Preprocessing - identical to SymptomDataPreprocessor in training
# ---------------------------------------------------------------------------


def preprocess_symptom_text(raw_text: str) -> str:
    """Cleans free-text symptom input using the exact same logic applied
    during training in SymptomDataPreprocessor.preprocess_df.

    Steps (kept in sync with training code):
      1. Strip surrounding whitespace.
      2. Lowercase conversion.
      3. Remove all characters that are not word chars, spaces, or underscores.
      4. Replace underscores with spaces.
      5. Collapse repeated whitespace and strip.

    Args:
        raw_text: Free-text symptom string from user.

    Returns:
        Cleaned symptom string ready for tokenization.
    """
    text = raw_text.strip()
    text = text.lower()
    text = re.sub(r"[^\w\s_]", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "no symptoms reported"


# ---------------------------------------------------------------------------
# BioBERT Inference Pipeline
# ---------------------------------------------------------------------------


class NLPInferencePipeline:
    """Production-grade inference pipeline for BioBERT Symptom Classifier.

    Loads the fine-tuned checkpoint, tokenizer, label encoder, and
    temperature scaler, and exposes a predict() method for symptom text.

    Attributes:
        device (torch.device): CPU or CUDA device.
        model: BertForSequenceClassification classifier in eval mode.
        tokenizer: AutoTokenizer loaded from saved artifacts.
        idx_to_disease (Dict[int, str]): Reverse mapping from label index to
            disease name.
        max_length (int): Token sequence length used during training (128).
        temperature (float): Confidence calibration temperature value.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = _DEFAULT_CHECKPOINT,
        tokenizer_dir: Union[str, Path] = _DEFAULT_TOKENIZER_DIR,
        disease_mapping_path: Union[str, Path] = _DEFAULT_DISEASE_MAPPING,
        model_name: str = _DEFAULT_MODEL_NAME,
        dropout: float = _DEFAULT_DROPOUT,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        """Initializes the NLP inference pipeline."""
        logger.info("Initializing NLP Inference Pipeline.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("NLP pipeline using device: %s", self.device)

        self.max_length = max_length

        # 1. Load Disease Mapping / Label Encoder
        label_encoder_path = Path(tokenizer_dir) / "label_encoder.pkl"
        if label_encoder_path.exists():
            try:
                with open(label_encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                self.idx_to_disease = {
                    idx: str(disease) for idx, disease in enumerate(self.label_encoder.classes_)
                }
                self.disease_to_idx = {
                    str(disease): idx for idx, disease in enumerate(self.label_encoder.classes_)
                }
                logger.info(
                    "Label encoder loaded. Re-aligned mapping with %d classes.",
                    len(self.idx_to_disease),
                )
            except Exception as e:
                logger.warning("Failed to load label encoder from %s: %s", label_encoder_path, e)
                self.disease_to_idx, self.idx_to_disease = self._load_disease_mapping(
                    Path(disease_mapping_path)
                )
        else:
            self.disease_to_idx, self.idx_to_disease = self._load_disease_mapping(
                Path(disease_mapping_path)
            )

        num_classes = len(self.disease_to_idx)
        logger.info("Disease mapping initialized: %d classes.", num_classes)

        # 2. Load Temperature Scaler
        temperature_scaler_path = Path(tokenizer_dir) / "temperature_scaler.json"
        self.temperature = 1.0
        if temperature_scaler_path.exists():
            try:
                with open(temperature_scaler_path, "r", encoding="utf-8") as f:
                    scaler_data = json.load(f)
                self.temperature = float(scaler_data.get("temperature", 1.0))
                logger.info("Temperature scaler loaded: T=%f", self.temperature)
            except Exception as e:
                logger.warning("Failed to load temperature scaler: %s", e)

        # 3. Reconstruct Model Architecture
        try:
            config = BertConfig.from_pretrained(
                model_name,
                num_labels=num_classes,
                seq_classif_dropout=dropout,
            )
            # Match vocabulary size of tokenizers/weights (30522)
            config.vocab_size = 30522
            inner_model = BertForSequenceClassification(config)
        except Exception as e:
            raise AppInferenceError(
                message=f"Failed to reconstruct BioBERT architecture: {e}",
                details={"model_name": model_name},
            )

        # 4. Load Checkpoint
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise AppInferenceError(
                message=f"NLP checkpoint not found: {ckpt_path}",
                details={"checkpoint_path": str(ckpt_path)},
            )

        try:
            raw = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise AppInferenceError(
                message=f"Failed to load NLP checkpoint: {e}",
                details={"checkpoint_path": str(ckpt_path)},
            )

        if isinstance(raw, dict) and "model_state_dict" in raw:
            saved_state: Dict[str, Any] = raw["model_state_dict"]
            epoch = raw.get("epoch", "N/A")
            metrics = raw.get("metrics", {})
        else:
            saved_state = raw
            epoch = "N/A"
            metrics = {}

        # Strip prefixes if model was saved inside a wrapper structure
        sample_key = next(iter(saved_state))
        if sample_key.startswith("bert."):
            logger.info("Stripping 'bert.' prefix from checkpoint keys.")
            saved_state = {k[5:]: v for k, v in saved_state.items()}
        elif sample_key.startswith("model."):
            logger.info("Stripping 'model.' prefix from checkpoint keys.")
            saved_state = {k[6:]: v for k, v in saved_state.items()}

        try:
            inner_model.load_state_dict(saved_state, strict=True)
        except RuntimeError as e:
            raise AppInferenceError(
                message=f"State dict mismatch loading NLP checkpoint: {e}",
                details={"checkpoint_path": str(ckpt_path)},
            )

        val_loss = metrics.get("val_loss", "N/A")
        val_acc = metrics.get("val_acc", "N/A")
        logger.info(
            "NLP checkpoint loaded successfully from: %s | num_params: %d",
            ckpt_path.name,
            sum(p.numel() for p in inner_model.parameters()),
        )
        print(f"[NLP] checkpoint: {ckpt_path.name}")
        print(f"[NLP] epoch:      {epoch}")
        print(f"[NLP] num_classes: {num_classes}")

        self.checkpoint_info = {
            "checkpoint_path": ckpt_path,
            "epoch": epoch,
            "metrics": metrics,
            "num_classes": num_classes,
        }

        # Put Model in Eval Mode
        inner_model.to(self.device)
        inner_model.eval()
        self.model = inner_model

        # 5. Load Tokenizer
        tokenizer_path = Path(tokenizer_dir)
        if not tokenizer_path.exists():
            raise AppInferenceError(
                message=f"Tokenizer directory not found: {tokenizer_path}",
                details={"tokenizer_dir": str(tokenizer_path)},
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
            )
            logger.info("Tokenizer loaded from local artifact dir: %s", tokenizer_path)
        except Exception as e:
            logger.warning(
                "Could not load tokenizer from local dir (%s). "
                "Falling back to HuggingFace Hub '%s': %s",
                tokenizer_path,
                model_name,
                e,
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info("Tokenizer loaded from HuggingFace Hub: %s", model_name)
            except Exception as e2:
                raise AppInferenceError(
                    message=(
                        f"Failed to load tokenizer from both local dir "
                        f"and HuggingFace Hub: {e2}"
                    ),
                    details={
                        "tokenizer_dir": str(tokenizer_path),
                        "model_name": model_name,
                    },
                )

        # 6. Load Clinical Explanations Registry
        explanations_path = Path(tokenizer_dir) / "clinical_explanations.json"
        self.explanations = {}
        if explanations_path.exists():
            try:
                with open(explanations_path, "r", encoding="utf-8") as f:
                    self.explanations = json.load(f)
                logger.info("Loaded clinical explanations registry from: %s", explanations_path)
            except Exception as e:
                logger.warning("Failed to load clinical explanations registry: %s", e)

        logger.info(
            "NLP Inference Pipeline ready. Classes: %d | MaxLen: %d | Device: %s",
            num_classes,
            max_length,
            self.device,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_disease_mapping(
        mapping_path: Path,
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Loads disease mapping JSON and builds the reverse index-to-name map."""
        if not mapping_path.exists():
            raise AppInferenceError(
                message=f"Disease mapping file not found: {mapping_path}",
                details={"path": str(mapping_path)},
            )
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                disease_to_idx: Dict[str, int] = json.load(f)
            idx_to_disease: Dict[int, str] = {v: k for k, v in disease_to_idx.items()}
            return disease_to_idx, idx_to_disease
        except Exception as e:
            raise AppInferenceError(
                message=f"Failed to load disease mapping: {e}",
                details={"path": str(mapping_path)},
            )

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes text using identical parameters to NLPTextDataset in training.

        Training parameters: padding=max_length, truncation=True,
        max_length=128, return_tensors=pt.

        Args:
            text: Preprocessed symptom string.

        Returns:
            Tuple of (input_ids, attention_mask) tensors of shape [1, max_length].
        """
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        return input_ids, attention_mask

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def predict(self, symptom_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Runs disease prediction from a free-text symptom description.

        Args:
            symptom_text: Natural-language symptom description.
                Example: "I have fever, cough, sore throat, headache and body pain."
            top_k: Number of top predictions to return (default: 5).

        Returns:
            Dict with keys:
                - predicted_disease (str): Top-1 predicted disease name.
                - confidence (float): Softmax probability of top-1 prediction.
                - top_predictions (List[Dict]): Top-k predictions each with
                  rank (int), disease (str), confidence (float).
                - preprocessed_text (str): The cleaned text that was tokenized.
                - clinical_explanation (Optional[Dict]): Explanations and guidance.

        Raises:
            AppValidationError: If symptom_text is empty after preprocessing.
            AppInferenceError: If inference fails.
        """
        if not isinstance(symptom_text, str) or not symptom_text.strip():
            raise AppValidationError(
                message="symptom_text must be a non-empty string.",
                details={"received": repr(symptom_text)},
            )

        logger.info("NLP inference on text: '%s'", symptom_text[:120])

        try:
            # 1. Preprocess - identical to training
            cleaned = preprocess_symptom_text(symptom_text)
            logger.debug("Preprocessed text: '%s'", cleaned)

            # 2. Tokenize
            input_ids, attention_mask = self._tokenize(cleaned)

            # 3. Inference
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # [1, num_classes]
                # Apply Temperature Scaling Confidence Calibration
                calibrated_logits = logits / self.temperature
                probabilities = F.softmax(calibrated_logits, dim=-1).squeeze(0)  # [num_classes]

            # 4. Top-k predictions
            k = min(top_k, len(self.idx_to_disease))
            top_probs, top_indices = torch.topk(probabilities, k=k)

            top_predictions: List[Dict[str, Any]] = []
            for rank, (prob, idx) in enumerate(
                zip(top_probs.tolist(), top_indices.tolist()), start=1
            ):
                disease_name = self.idx_to_disease.get(idx, f"Unknown({idx})")
                top_predictions.append(
                    {
                        "rank": rank,
                        "disease": disease_name,
                        "confidence": round(prob, 6),
                    }
                )

            best = top_predictions[0]
            predicted_name = best["disease"]

            # Fetch clinical explanation details
            explanation = self.explanations.get(predicted_name, None)

            result: Dict[str, Any] = {
                "predicted_disease": predicted_name,
                "confidence": best["confidence"],
                "top_predictions": top_predictions,
                "preprocessed_text": cleaned,
                "clinical_explanation": explanation,
            }

            logger.info(
                "NLP inference complete. Predicted: %s (confidence=%.4f)",
                best["disease"],
                best["confidence"],
            )
            return result

        except (AppValidationError, AppInferenceError):
            raise
        except Exception as e:
            raise AppInferenceError(
                message=f"NLP inference execution failed: {e}",
                details={"symptom_text": symptom_text[:200]},
            )
