"""
Print basic statistics about the CLINC150 "plus" dataset and the vocabulary
that would be built from its training split.

Usage: python scripts/dataset_stats.py
"""

import sys
from pathlib import Path

# Allow running as `python scripts/dataset_stats.py` from the repo root
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MIN_FREQ
from src.data import load_clinc
from src.text_preprocessing import Vocab, tokenize


def main() -> None:
    train, val, test, id2label, label2id = load_clinc()

    print("Split sizes:")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}")
    print(f"  test:  {len(test)}")

    print(f"\nNumber of labels: {len(id2label)}")

    train_texts = [text for text, _ in train]
    vocab = Vocab.build(train_texts, min_freq=MIN_FREQ)
    print(f"\nVocab size (min_freq={MIN_FREQ}, built on train split): {len(vocab)}")

    val_texts = [text for text, _ in val]
    total_tokens = 0
    oov_tokens = 0
    for text in val_texts:
        for token in tokenize(text):
            total_tokens += 1
            if token not in vocab.token_to_id:
                oov_tokens += 1

    oov_rate = oov_tokens / total_tokens if total_tokens else 0.0
    print(f"\nValidation OOV token rate: {oov_rate:.4f} ({oov_tokens}/{total_tokens} tokens)")


if __name__ == "__main__":
    main()
