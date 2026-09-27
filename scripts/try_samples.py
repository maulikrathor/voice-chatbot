"""
Run the full VoiceChatbot pipeline over every audio file in samples/ and
print a table: file, transcript, intent, confidence, response, expected,
match. Also saves the table to artifacts/sample_results.md for the report.

Sample files are named NN_<expected_intent>.wav (e.g. 04_book_flight.wav).
"silence" is a special expected value meaning the file is expected to be
rejected by the ASR gate (status: silent / too_short / no_speech), not
classified as an intent.

Usage: python scripts/try_samples.py
"""

import json
import sys
from pathlib import Path

# Allow running as `python scripts/try_samples.py` from the repo root
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa

from src.config import ARTIFACTS_DIR, LABELS_FILENAME, SAMPLES_DIR
from src.pipeline import VoiceChatbot

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")

# ASR statuses that count as "rejected" for a file whose expected intent is
# "silence" (see src.pipeline._ASR_STATUSES_TO_CLASSIFY for the complement).
_ASR_REJECTION_STATUSES = ("silent", "too_short", "no_speech")

RESULTS_MD_PATH = ARTIFACTS_DIR / "sample_results.md"


def _expected_intent_from_filename(path: Path) -> str:
    """Everything after the first underscore, without the extension."""
    return path.stem.split("_", 1)[1]


def main() -> None:
    sample_files = (
        sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
        if SAMPLES_DIR.exists()
        else []
    )

    if not sample_files:
        print(f"No audio files found in {SAMPLES_DIR}. Add .wav/.flac/.ogg/.mp3 files and rerun.")
        sys.exit(0)

    with (ARTIFACTS_DIR / LABELS_FILENAME).open("r", encoding="utf-8") as f:
        labels = set(json.load(f))

    # Sanity-check filenames against labels.json before running anything
    # expensive; "silence" is not a label, it's a special expected outcome.
    unknown_expected = sorted(
        {
            _expected_intent_from_filename(p)
            for p in sample_files
            if _expected_intent_from_filename(p) != "silence" and _expected_intent_from_filename(p) not in labels
        }
    )
    if unknown_expected:
        print(f"WARNING: expected intents not found in labels.json: {unknown_expected}")
    else:
        print("All expected intents in sample filenames exist in labels.json.")

    print("Loading VoiceChatbot pipeline (this loads Whisper, may take a moment)...")
    chatbot = VoiceChatbot(load_asr=True)

    rows = []
    for path in sample_files:
        expected = _expected_intent_from_filename(path)

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

        if expected == "silence":
            matched = result["asr_status"] in _ASR_REJECTION_STATUSES
            actual = result["asr_status"]
        else:
            matched = result["intent"] == expected
            actual = result["intent"]

        rows.append(
            {
                "file": path.name,
                "transcript": result["transcript"],
                "intent": result["intent"],
                "confidence": result["confidence"],
                "response": result["response"],
                "expected": expected,
                "actual": actual,
                "match": "Y" if matched else "N",
            }
        )

    if not rows:
        print("No audio files could be decoded.")
        sys.exit(0)

    header = f"{'file':<20} {'transcript':<35} {'intent':<20} {'conf':<7} {'expected':<18} {'actual':<18} match"
    print(f"\n{header}")
    for r in rows:
        print(
            f"{r['file']:<20} {r['transcript'][:35]:<35} {r['intent']:<20} "
            f"{r['confidence']:<7.3f} {r['expected']:<18} {r['actual']:<18} {r['match']}"
        )

    num_matched = sum(1 for r in rows if r["match"] == "Y")
    print(f"\n{num_matched}/{len(rows)} matched")

    _write_results_md(rows, num_matched)
    print(f"Saved results table to {RESULTS_MD_PATH}")


def _write_results_md(rows: list[dict], num_matched: int) -> None:
    lines = [
        "# Sample results",
        "",
        "| file | transcript | intent | confidence | response | expected | actual | match |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['file']} | {r['transcript']} | {r['intent']} | {r['confidence']:.3f} | "
            f"{r['response']} | {r['expected']} | {r['actual']} | {r['match']} |"
        )
    lines.append("")
    lines.append(f"**{num_matched}/{len(rows)} matched**")
    lines.append("")

    RESULTS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
