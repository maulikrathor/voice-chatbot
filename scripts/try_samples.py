"""
Run the full VoiceChatbot pipeline over every audio file in samples/ and
print a table: file, transcript, intent, confidence, response.

Usage: python scripts/try_samples.py
"""

import sys
from pathlib import Path

# Allow running as `python scripts/try_samples.py` from the repo root
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa

from src.config import REPO_ROOT
from src.pipeline import VoiceChatbot

SAMPLES_DIR = REPO_ROOT / "samples"
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")


def main() -> None:
    sample_files = (
        sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
        if SAMPLES_DIR.exists()
        else []
    )

    if not sample_files:
        print(f"No audio files found in {SAMPLES_DIR}. Add .wav/.flac/.ogg/.mp3 files and rerun.")
        sys.exit(0)

    print("Loading VoiceChatbot pipeline (this loads Whisper, may take a moment)...")
    chatbot = VoiceChatbot(load_asr=True)

    rows = []
    for path in sample_files:
        try:
            # sr=None keeps the file's native sample rate; the pipeline's
            # own audio prep step handles resampling to 16 kHz.
            data, sample_rate = librosa.load(path, sr=None, mono=False)
        except Exception as exc:
            print(f"Skipping {path.name}: failed to decode ({exc})")
            continue

        # librosa.load returns (channels, samples) for stereo, (samples,)
        # for mono; VoiceChatbot/prepare_audio expects (samples,) or
        # (samples, channels).
        if data.ndim == 2:
            data = data.T

        result = chatbot.handle_audio(sample_rate, data)
        rows.append(
            (path.name, result["transcript"], result["intent"], result["confidence"], result["response"])
        )

    if not rows:
        print("No audio files could be decoded.")
        sys.exit(0)

    print(f"\n{'file':<25} {'transcript':<40} {'intent':<20} {'confidence':<10} response")
    for name, transcript, intent, confidence, response in rows:
        print(f"{name:<25} {transcript[:40]:<40} {intent:<20} {confidence:<10.3f} {response}")


if __name__ == "__main__":
    main()
