"""
Text preprocessing shared by both training and inference.

This module is the single source of truth for turning raw text into model
input ids. `train.py` builds the Vocab from the training split and saves it;
`predict.py` (and, indirectly, `pipeline.py`) load the same Vocab and call
the same `normalize_text` / `tokenize` functions, so a transcript is
preprocessed identically at inference time as every training example was.
"""

import json
import re
from collections import Counter
from pathlib import Path

from src.config import MAX_LEN, MIN_FREQ, PAD_ID, PAD_TOKEN, UNK_ID, UNK_TOKEN

# Matches any character that is not a lowercase letter, digit, apostrophe,
# or whitespace. Used to strip punctuation while keeping contractions like
# "what's" intact.
_DISALLOWED_CHARS_RE = re.compile(r"[^a-z0-9' ]")

# Matches one or more whitespace characters, used to collapse runs of
# spaces/tabs/newlines left behind after stripping punctuation.
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Normalize raw text into a canonical lowercase form.

    Steps: lowercase, drop everything except [a-z0-9'] and spaces, collapse
    runs of whitespace into a single space, strip leading/trailing spaces.

    Example: "What's the  WEATHER?!" -> "what's the weather"
    """
    lowered = text.lower()
    stripped_punct = _DISALLOWED_CHARS_RE.sub("", lowered)
    collapsed = _WHITESPACE_RE.sub(" ", stripped_punct)
    return collapsed.strip()


def tokenize(text: str) -> list[str]:
    """Normalize text and split it into whitespace-separated tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


class Vocab:
    """
    A simple word-level vocabulary mapping tokens to integer ids.

    Ids 0 and 1 are always reserved for the pad and unk tokens
    (see src.config.PAD_ID / UNK_ID) so that model.py can rely on a fixed
    padding_idx and every Vocab instance is compatible with every other.
    """

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = MIN_FREQ) -> "Vocab":
        """
        Build a vocabulary from a list of raw texts.

        Tokens are counted after tokenize(); only tokens occurring at least
        `min_freq` times are kept, so rarer words fall back to <unk> both
        during training and at inference time. Kept tokens are assigned ids
        in descending frequency order (ties broken alphabetically) starting
        at 2, since 0/1 are reserved for <pad>/<unk>.
        """
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))

        kept_tokens = [token for token, count in counts.items() if count >= min_freq]
        kept_tokens.sort(key=lambda token: (-counts[token], token))

        token_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for token in kept_tokens:
            token_to_id[token] = len(token_to_id)

        return cls(token_to_id)

    def encode(self, text: str, max_len: int = MAX_LEN) -> list[int]:
        """
        Convert text into a fixed-length list of token ids.

        Unknown tokens map to UNK_ID. Sequences shorter than max_len are
        right-padded with PAD_ID; longer sequences are truncated to max_len.
        """
        tokens = tokenize(text)
        ids = [self.token_to_id.get(token, UNK_ID) for token in tokens]

        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [PAD_ID] * (max_len - len(ids))

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, path: Path) -> None:
        """Save the token-to-id mapping as JSON."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        """Load a vocabulary previously written by Vocab.save()."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        return cls(token_to_id)
