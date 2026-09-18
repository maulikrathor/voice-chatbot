"""
Training script for the BiLSTM intent classifier.

Loads CLINC150 plus via src.data, builds the Vocab from the training split
only, trains BiLSTMClassifier with early stopping on validation accuracy,
tunes the OOS confidence threshold on the validation split, and saves
model.pt, vocab.json, labels.json, config.json, and training_history.json
to artifacts/.

Run as: python -m src.train
"""

import contextlib
import io
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    CONFIG_FILENAME,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EMBED_DIM,
    GRAD_CLIP_NORM,
    HIDDEN_DIM,
    LABELS_FILENAME,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_LEN,
    MIN_FREQ,
    MODEL_FILENAME,
    PAD_ID,
    SEED,
    THRESHOLD_CANDIDATES,
    TRAINING_HISTORY_FILENAME,
    VOCAB_FILENAME,
)
from src.data import load_clinc
from src.model import BiLSTMClassifier
from src.text_preprocessing import Vocab, tokenize

DEVICE = torch.device("cpu")  # CPU-only per PROJECT_SPEC.md / ARCHITECTURE.md.


def set_seed(seed: int) -> None:
    """Seed every RNG the training run touches, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IntentDataset(Dataset):
    """Pre-encodes (text, label_id) pairs into fixed-length id/length/label tensors."""

    def __init__(self, pairs: list[tuple[str, int]], vocab: Vocab, max_len: int):
        self.token_ids = []
        self.lengths = []
        self.labels = []
        for text, label_id in pairs:
            self.token_ids.append(vocab.encode(text, max_len=max_len))
            # True (pre-padding) length, clamped to at least 1 so the model
            # never has to pack a zero-length sequence.
            true_len = max(min(len(tokenize(text)), max_len), 1)
            self.lengths.append(true_len)
            self.labels.append(label_id)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.token_ids[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def _load_dataset_with_source():
    """
    Call load_clinc() while capturing the "[data] Loaded ..." message it
    prints, so config.json can record which loading path (HF dataset id or
    GitHub fallback) was actually used -- without changing src/data.py's
    return signature.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        train_pairs, val_pairs, test_pairs, id2label, label2id = load_clinc()
    log_output = buffer.getvalue()
    print(log_output, end="")

    dataset_source = "unknown"
    for line in log_output.splitlines():
        if line.startswith("[data] "):
            dataset_source = line.removeprefix("[data] ").strip()

    return train_pairs, val_pairs, test_pairs, id2label, label2id, dataset_source


def _run_epoch(model, loader, optimizer, criterion):
    """One training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for token_ids, lengths, labels in loader:
        token_ids, lengths, labels = (
            token_ids.to(DEVICE),
            lengths.to(DEVICE),
            labels.to(DEVICE),
        )

        optimizer.zero_grad()
        logits = model(token_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate_accuracy(model, loader) -> float:
    """Plain argmax accuracy over a loader (no threshold applied)."""
    model.eval()
    correct = 0
    total = 0
    for token_ids, lengths, labels in loader:
        token_ids, lengths, labels = (
            token_ids.to(DEVICE),
            lengths.to(DEVICE),
            labels.to(DEVICE),
        )
        logits = model(token_ids, lengths)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def _predict_probs(model, loader):
    """Run the model over a loader, returning (softmax_probs (N,C), true_labels (N,)) as numpy arrays."""
    model.eval()
    all_probs = []
    all_labels = []
    for token_ids, lengths, labels in loader:
        token_ids, lengths = token_ids.to(DEVICE), lengths.to(DEVICE)
        logits = model(token_ids, lengths)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def _tune_threshold(probs: np.ndarray, true_labels: np.ndarray, oos_label_id: int):
    """
    Search THRESHOLD_CANDIDATES for the threshold t that maximizes overall
    accuracy when predictions with max softmax probability < t are
    relabeled as oos. Ties are broken in favor of the lower threshold
    (candidates are tried in ascending order and only strictly better
    accuracy replaces the current best).
    """
    max_probs = probs.max(axis=1)
    raw_preds = probs.argmax(axis=1)

    best_threshold = THRESHOLD_CANDIDATES[0]
    best_acc = -1.0
    for t in THRESHOLD_CANDIDATES:
        preds = np.where(max_probs < t, oos_label_id, raw_preds)
        acc = (preds == true_labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return float(best_threshold), float(best_acc)


def main():
    start_time = time.time()
    set_seed(SEED)

    print("Loading CLINC150 plus dataset...")
    train_pairs, val_pairs, _test_pairs, id2label, label2id, dataset_source = (
        _load_dataset_with_source()
    )
    # The oos label id is NOT assumed to be the last class -- it is looked
    # up by name, since the HF ClassLabel ordering places it at index 42.
    oos_label_id = label2id["oos"]
    num_classes = len(id2label)

    print("\nBuilding vocabulary from the train split only...")
    train_texts = [text for text, _ in train_pairs]
    vocab = Vocab.build(train_texts, min_freq=MIN_FREQ)
    print(f"Vocab size: {len(vocab)}")

    train_dataset = IntentDataset(train_pairs, vocab, MAX_LEN)
    val_dataset = IntentDataset(val_pairs, vocab, MAX_LEN)

    # num_workers=0: multi-worker DataLoader workers require the __main__
    # guard/spawn dance on Windows; 0 keeps this simple and reliable there.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        num_classes=num_classes,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        pad_idx=PAD_ID,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0
    history = []

    print(f"\nTraining for up to {MAX_EPOCHS} epochs (patience={EARLY_STOPPING_PATIENCE})...\n")
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = _run_epoch(model, train_loader, optimizer, criterion)
        val_acc = _evaluate_accuracy(model, val_loader)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:2d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | time={epoch_time:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping: no val_acc improvement for "
                    f"{EARLY_STOPPING_PATIENCE} epochs."
                )
                break

    print(f"\nBest epoch: {best_epoch} (val_acc={best_val_acc:.4f})")
    model.load_state_dict(best_state_dict)

    print("\nTuning OOS confidence threshold on the validation split...")
    val_probs, val_labels = _predict_probs(model, val_loader)
    threshold, threshold_val_acc = _tune_threshold(val_probs, val_labels, oos_label_id)
    print(f"Tuned threshold: {threshold:.2f} (val accuracy with threshold: {threshold_val_acc:.4f})")

    # --- Save artifacts -----------------------------------------------

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / MODEL_FILENAME
    torch.save(model.state_dict(), model_path)

    vocab_path = ARTIFACTS_DIR / VOCAB_FILENAME
    vocab.save(vocab_path)

    labels_path = ARTIFACTS_DIR / LABELS_FILENAME
    id2label_list = [id2label[i] for i in range(num_classes)]
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(id2label_list, f, ensure_ascii=False, indent=2)

    config_path = ARTIFACTS_DIR / CONFIG_FILENAME
    saved_config = {
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "pad_id": PAD_ID,
        "max_len": MAX_LEN,
        "min_freq": MIN_FREQ,
        "vocab_size": len(vocab),
        "num_classes": num_classes,
        "oos_label_id": oos_label_id,
        "threshold": threshold,
        "seed": SEED,
        "best_epoch": best_epoch,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "dataset_source": dataset_source,
        "torch_version": torch.__version__,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(saved_config, f, ensure_ascii=False, indent=2)

    history_path = ARTIFACTS_DIR / TRAINING_HISTORY_FILENAME
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s")

    print("\nArtifact sizes:")
    for path in (model_path, vocab_path, labels_path, config_path, history_path):
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
