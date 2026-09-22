"""
Evaluation script for the trained BiLSTM intent classifier.

Loads ONLY the artifacts saved by train.py (model.pt, vocab.json,
labels.json, config.json) -- the vocabulary and every model hyperparameter
come from those files, not from src.config, which proves the saved
artifacts are self-sufficient for inference. The CLINC150 plus test split
itself is fetched via src.data (it is raw evaluation data, not a trained
artifact).

Run as: python -m src.evaluate
"""

import json
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import (
    ARTIFACTS_DIR,
    CONFIG_FILENAME,
    LABELS_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    VOCAB_FILENAME,
)
from src.data import load_clinc
from src.model import BiLSTMClassifier
from src.text_preprocessing import Vocab, tokenize


def _load_artifacts():
    """Load model, vocab, labels, and hyperparameters strictly from artifacts/."""
    with (ARTIFACTS_DIR / CONFIG_FILENAME).open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    with (ARTIFACTS_DIR / LABELS_FILENAME).open("r", encoding="utf-8") as f:
        id2label_list = json.load(f)

    vocab = Vocab.load(ARTIFACTS_DIR / VOCAB_FILENAME)

    model = BiLSTMClassifier(
        vocab_size=model_config["vocab_size"],
        num_classes=model_config["num_classes"],
        embed_dim=model_config["embed_dim"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
        pad_idx=model_config["pad_id"],
    )
    state_dict = torch.load(ARTIFACTS_DIR / MODEL_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, vocab, id2label_list, model_config


def _encode_split(pairs, vocab: Vocab, max_len: int):
    """Encode a (text, label_id) split into token id / length tensors and a label array."""
    token_ids = []
    lengths = []
    labels = []
    for text, label_id in pairs:
        token_ids.append(vocab.encode(text, max_len=max_len))
        true_len = max(min(len(tokenize(text)), max_len), 1)
        lengths.append(true_len)
        labels.append(label_id)

    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        np.array(labels),
    )


@torch.no_grad()
def _predict_probs(model, token_ids: torch.Tensor, lengths: torch.Tensor, batch_size: int = 64):
    """Run the model over the split in batches, returning softmax probs (N, C)."""
    all_probs = []
    for start in range(0, token_ids.size(0), batch_size):
        batch_ids = token_ids[start : start + batch_size]
        batch_lengths = lengths[start : start + batch_size]
        logits = model(batch_ids, batch_lengths)
        all_probs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(all_probs, axis=0)


def _compute_metrics(true_labels: np.ndarray, pred_labels: np.ndarray, oos_label_id: int, num_classes: int):
    """Compute in-scope accuracy, OOS recall/precision, overall accuracy, and macro-F1."""
    in_scope_mask = true_labels != oos_label_id
    in_scope_acc = (
        (pred_labels[in_scope_mask] == true_labels[in_scope_mask]).mean()
        if in_scope_mask.any()
        else float("nan")
    )
    overall_acc = (pred_labels == true_labels).mean()

    is_true_oos = true_labels == oos_label_id
    is_pred_oos = pred_labels == oos_label_id
    oos_recall = recall_score(is_true_oos, is_pred_oos, pos_label=True, zero_division=0)
    oos_precision = precision_score(is_true_oos, is_pred_oos, pos_label=True, zero_division=0)

    macro_f1 = f1_score(
        true_labels,
        pred_labels,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    # Same objective train.py's tune_threshold() optimizes for -- reported
    # here too so the test-set number can be compared directly to the
    # validation number the threshold was picked on.
    balanced_acc = 0.5 * (in_scope_acc + oos_recall)

    return {
        "in_scope_accuracy": float(in_scope_acc),
        "oos_recall": float(oos_recall),
        "oos_precision": float(oos_precision),
        "overall_accuracy": float(overall_acc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
    }


def _top_confused_pairs(true_labels: np.ndarray, pred_labels: np.ndarray, id2label_list: list[str], top_k: int = 10):
    """Return the top_k most frequent (true_label_name, predicted_label_name) mismatches."""
    confusions = Counter(
        (id2label_list[t], id2label_list[p])
        for t, p in zip(true_labels, pred_labels)
        if t != p
    )
    return confusions.most_common(top_k)


def _print_metrics(metrics: dict) -> None:
    print(f"  in-scope accuracy: {metrics['in_scope_accuracy']:.4f}")
    print(f"  oos recall:        {metrics['oos_recall']:.4f}")
    print(f"  oos precision:     {metrics['oos_precision']:.4f}")
    print(f"  overall accuracy:  {metrics['overall_accuracy']:.4f}")
    print(f"  macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"  balanced accuracy: {metrics['balanced_accuracy']:.4f}")


def main():
    print("Loading artifacts from artifacts/ ...")
    model, vocab, id2label_list, model_config = _load_artifacts()
    num_classes = model_config["num_classes"]
    oos_label_id = model_config["oos_label_id"]
    threshold = model_config["threshold"]
    max_len = model_config["max_len"]

    print("Loading CLINC150 plus test split...")
    _train_pairs, _val_pairs, test_pairs, _id2label, _label2id = load_clinc()

    token_ids, lengths, true_labels = _encode_split(test_pairs, vocab, max_len)
    probs = _predict_probs(model, token_ids, lengths)
    raw_preds = probs.argmax(axis=1)

    max_probs = probs.max(axis=1)
    thresholded_preds = np.where(max_probs < threshold, oos_label_id, raw_preds)

    metrics_without_threshold = _compute_metrics(true_labels, raw_preds, oos_label_id, num_classes)
    metrics_with_threshold = _compute_metrics(true_labels, thresholded_preds, oos_label_id, num_classes)
    top_confused = _top_confused_pairs(true_labels, thresholded_preds, id2label_list)

    results = {
        "threshold": threshold,
        "without_threshold": metrics_without_threshold,
        "with_threshold": metrics_with_threshold,
        "top_confused_pairs": [
            {"true": t, "predicted": p, "count": c} for (t, p), c in top_confused
        ],
    }

    metrics_path = ARTIFACTS_DIR / METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== Test metrics (WITHOUT threshold, argmax only) ===")
    _print_metrics(metrics_without_threshold)

    print(f"\n=== Test metrics (WITH tuned threshold={threshold:.2f}) ===")
    _print_metrics(metrics_with_threshold)

    print("\n=== Top 10 confused (true -> predicted) pairs ===")
    for (true_name, pred_name), count in top_confused:
        print(f"  {true_name} -> {pred_name}: {count}")

    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
