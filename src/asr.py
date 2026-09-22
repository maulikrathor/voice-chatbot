"""
Speech-to-text via Whisper base.en.

SpeechRecognizer wraps a Hugging Face "automatic-speech-recognition"
pipeline (openai/whisper-base.en) with an audio-preparation step and a
silence/length gate. Whisper hallucinates plausible-looking text (e.g.
"Thank you.") when given silence or near-silence, so the gate rejects
those clips before they ever reach the model, rather than trusting
whatever Whisper returns.
"""

import numpy as np
import librosa
from transformers import pipeline

from src.config import (
    MAX_AUDIO_DURATION_SEC,
    MIN_AUDIO_DURATION_SEC,
    SILENCE_RMS_THRESHOLD,
    TARGET_SAMPLE_RATE,
    WHISPER_MODEL_ID,
)


def prepare_audio(sample_rate: int, data: np.ndarray) -> np.ndarray:
    """
    Convert raw (sample_rate, data) audio -- as returned by Gradio's mic/
    upload components -- into float32 mono audio at TARGET_SAMPLE_RATE.

    Pure function: does not touch the Whisper model, so it can be unit
    tested with plain numpy arrays.

    Handles:
      - int16 / int32 input: scaled by the dtype's max magnitude into
        roughly [-1, 1].
      - float input: assumed already roughly in [-1, 1], passed through.
      - stereo input of shape (N, 2): averaged to mono.
      - any sample_rate != TARGET_SAMPLE_RATE: resampled via librosa.
    """
    audio = np.asarray(data)

    if np.issubdtype(audio.dtype, np.integer):
        max_magnitude = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_magnitude
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 2:
        # (N, channels) -> mono by averaging channels.
        audio = audio.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

    return audio.astype(np.float32)


class SpeechRecognizer:
    """Loads the Whisper base.en ASR pipeline once; transcribe() applies the silence/length gate first."""

    def __init__(self, model_id: str = WHISPER_MODEL_ID):
        # device=-1 => CPU, matching the project's CPU-only deployment target.
        self._asr_pipeline = pipeline("automatic-speech-recognition", model=model_id, device=-1)

    def transcribe(self, sample_rate: int, data: np.ndarray) -> dict:
        """
        Prepare audio, apply the silence/length gate, and transcribe.

        Returns {text, duration_s, status}, where status is one of:
            "ok", "too_short", "silent", "truncated", "error", "no_speech".
        """
        try:
            audio = prepare_audio(sample_rate, data)
        except Exception:
            return {"text": "", "duration_s": 0.0, "status": "error"}

        duration_s = len(audio) / TARGET_SAMPLE_RATE

        if duration_s < MIN_AUDIO_DURATION_SEC:
            return {"text": "", "duration_s": duration_s, "status": "too_short"}

        # RMS gate: Whisper hallucinates plausible text like "Thank you."
        # on silence/near-silence rather than returning nothing, so silent
        # clips must be rejected here -- checking the transcript alone
        # would not catch this, since Whisper would still return *something*.
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        if rms < SILENCE_RMS_THRESHOLD:
            return {"text": "", "duration_s": duration_s, "status": "silent"}

        status = "ok"
        if duration_s > MAX_AUDIO_DURATION_SEC:
            audio = audio[: int(MAX_AUDIO_DURATION_SEC * TARGET_SAMPLE_RATE)]
            duration_s = len(audio) / TARGET_SAMPLE_RATE
            status = "truncated"

        try:
            result = self._asr_pipeline({"raw": audio, "sampling_rate": TARGET_SAMPLE_RATE})
            text = result["text"].strip()
        except Exception:
            return {"text": "", "duration_s": duration_s, "status": "error"}

        if not text:
            return {"text": "", "duration_s": duration_s, "status": "no_speech"}

        return {"text": text, "duration_s": duration_s, "status": status}
