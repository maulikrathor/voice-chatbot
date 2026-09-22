"""
Inference wrapper around the trained BiLSTM intent classifier.

IntentPredictor loads the artifacts saved by train.py (model.pt,
vocab.json, labels.json, config.json) once, and exposes predict(text),
which reuses the exact same normalize_text / tokenize / Vocab.encode
pipeline used during training (src.text_preprocessing) so a transcript is
preprocessed identically at inference time as every training example was.
"""

import json
from pathlib import Path

import torch

from src.config import ARTIFACTS_DIR, CONFIG_FILENAME, LABELS_FILENAME, MODEL_FILENAME, VOCAB_FILENAME
from src.model import BiLSTMClassifier
from src.text_preprocessing import Vocab, normalize_text, tokenize

# Number of top predictions reported alongside the chosen intent.
TOP_K = 3


class IntentPredictor:
    """Loads BiLSTM artifacts once; predict(text) returns intent + confidence + fallback info."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        artifacts_dir = Path(artifacts_dir)

        with (artifacts_dir / CONFIG_FILENAME).open("r", encoding="utf-8") as f:
            self.config: dict = json.load(f)

        with (artifacts_dir / LABELS_FILENAME).open("r", encoding="utf-8") as f:
            self.id2label: list[str] = json.load(f)

        self.vocab = Vocab.load(artifacts_dir / VOCAB_FILENAME)

        self.max_len: int = self.config["max_len"]
        self.oos_label_id: int = self.config["oos_label_id"]
        self.threshold: float = self.config["threshold"]

        self.model = BiLSTMClassifier(
            vocab_size=self.config["vocab_size"],
            num_classes=self.config["num_classes"],
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            dropout=self.config["dropout"],
            pad_idx=self.config["pad_id"],
        )
        state_dict = torch.load(artifacts_dir / MODEL_FILENAME, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Predict the intent for a piece of raw input text.

        Returns a dict with keys:
            input_text: the raw text passed in.
            normalized_text: text after src.text_preprocessing.normalize_text.
            intent: final intent name (may be "oos" due to a fallback rule).
            confidence: softmax probability of the raw top-1 prediction.
            top3: list of (label, prob) for the top TOP_K raw predictions
                (empty for the empty_text fallback, since there is nothing
                to run the model on).
            is_fallback: True if a fallback rule fired.
            fallback_reason: None, "empty_text", "predicted_oos", or
                "low_confidence".
        """
        normalized = normalize_text(text)

        if not normalized:
            return {
                "input_text": text,
                "normalized_text": normalized,
                "intent": "oos",
                "confidence": 0.0,
                "top3": [],
                "is_fallback": True,
                "fallback_reason": "empty_text",
            }

        token_ids = self.vocab.encode(normalized, max_len=self.max_len)
        true_len = max(min(len(tokenize(normalized)), self.max_len), 1)

        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)
        lengths_tensor = torch.tensor([true_len], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(token_ids_tensor, lengths_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

        top_probs, top_ids = torch.topk(probs, k=min(TOP_K, probs.size(0)))
        top3 = [(self.id2label[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_ids)]

        raw_label_id = int(top_ids[0].item())
        raw_confidence = float(top_probs[0].item())
        raw_intent = self.id2label[raw_label_id]

        if raw_label_id == self.oos_label_id:
            return {
                "input_text": text,
                "normalized_text": normalized,
                "intent": "oos",
                "confidence": raw_confidence,
                "top3": top3,
                "is_fallback": True,
                "fallback_reason": "predicted_oos",
            }

        if raw_confidence < self.threshold:
            # top3 still reports the raw top-1 intent (it's already in
            # there); only the *reported* intent is overridden to "oos"
            # since we don't trust a below-threshold prediction.
            return {
                "input_text": text,
                "normalized_text": normalized,
                "intent": "oos",
                "confidence": raw_confidence,
                "top3": top3,
                "is_fallback": True,
                "fallback_reason": "low_confidence",
            }

        return {
            "input_text": text,
            "normalized_text": normalized,
            "intent": raw_intent,
            "confidence": raw_confidence,
            "top3": top3,
            "is_fallback": False,
            "fallback_reason": None,
        }
