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

# --- Model hyperparameters (BiLSTMClassifier, ARCHITECTURE.md section 4) ---

EMBED_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.3

# --- Training hyperparameters --------------------------------------------

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
GRAD_CLIP_NORM = 1.0

# Candidate OOS confidence thresholds tried during threshold tuning on the
# validation split: 0.00, 0.05, ..., 0.95.
THRESHOLD_CANDIDATES = [round(i * 0.05, 2) for i in range(20)]

# --- Artifact filenames (artifacts/) ---------------------------------------

MODEL_FILENAME = "model.pt"
VOCAB_FILENAME = "vocab.json"
LABELS_FILENAME = "labels.json"
CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.json"
TRAINING_HISTORY_FILENAME = "training_history.json"

# --- ASR (src/asr.py) -------------------------------------------------------

# Hugging Face model id for the Whisper ASR pipeline (ARCHITECTURE.md section 4).
WHISPER_MODEL_ID = "openai/whisper-base.en"

# All audio is prepared to mono float32 at this sample rate before
# transcription, matching what Whisper expects.
TARGET_SAMPLE_RATE = 16000

# Silence/length gate thresholds. Whisper hallucinates plausible-looking
# text on silence/near-silence rather than returning nothing, so clips
# below these thresholds are rejected before ever reaching the model.
MIN_AUDIO_DURATION_SEC = 0.5
MAX_AUDIO_DURATION_SEC = 30.0
SILENCE_RMS_THRESHOLD = 0.01
