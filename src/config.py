"""
Central configuration for the voice chatbot project.

Holds filesystem paths and the hyperparameters/constants that must stay
identical between training (train.py) and inference (predict.py, pipeline.py)
so that preprocessing and model I/O never drift apart.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

# Repository root (this file lives at <repo_root>/src/config.py).
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# --- Reproducibility ----------------------------------------------------

SEED = 42

# --- Text preprocessing / vocabulary ------------------------------------

# Maximum number of tokens kept per example; shorter sequences are padded,
# longer ones are truncated.
MAX_LEN = 32

# Minimum number of occurrences (in the training split) for a token to get
# its own vocabulary entry. Rarer tokens map to the <unk> id.
MIN_FREQ = 2

# Reserved vocabulary tokens and their fixed ids.
PAD_TOKEN = "<pad>"
PAD_ID = 0
UNK_TOKEN = "<unk>"
UNK_ID = 1

# --- Classification -------------------------------------------------------

# CLINC150 "plus" split: 150 in-scope intents + 1 out-of-scope ("oos") class.
NUM_CLASSES = 151
