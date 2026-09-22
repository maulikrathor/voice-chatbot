"""
End-to-end voice chatbot pipeline.

VoiceChatbot combines SpeechRecognizer, IntentPredictor, and
ResponseEngine to turn either raw audio or typed text into a single result
dict. This module only orchestrates those three -- no preprocessing,
model, or response logic is duplicated here.
"""

from pathlib import Path

from src.asr import SpeechRecognizer
from src.config import ARTIFACTS_DIR, DATA_DIR
from src.predict import IntentPredictor
from src.responses import ASR_STATUS_MESSAGES, ResponseEngine

# ASR statuses that still warrant running the classifier. Every other
# status (too_short, silent, no_speech, error) means the gate rejected
# the clip, so we skip the classifier and return the matching system
# message instead.
_ASR_STATUSES_TO_CLASSIFY = ("ok", "truncated")


class VoiceChatbot:
    """Wires SpeechRecognizer + IntentPredictor + ResponseEngine into one pipeline."""

    def __init__(
        self,
        artifacts_dir: Path = ARTIFACTS_DIR,
        data_dir: Path = DATA_DIR,
        load_asr: bool = True,
    ):
        self.predictor = IntentPredictor(artifacts_dir)
        self.responder = ResponseEngine(data_dir=data_dir, artifacts_dir=artifacts_dir)
        # Whisper is the slow/heavy part to load; text-only callers (and
        # tests) can skip it entirely via load_asr=False.
        self.recognizer = SpeechRecognizer() if load_asr else None

    def handle_text(self, text: str) -> dict:
        """Classify typed text and generate a response."""
        prediction = self.predictor.predict(text)
        response = self.responder.respond(prediction["intent"])

        return {
            "source": "text",
            "transcript": text,
            "intent": prediction["intent"],
            "confidence": prediction["confidence"],
            "top3": prediction["top3"],
            "is_fallback": prediction["is_fallback"],
            "fallback_reason": prediction["fallback_reason"],
            "response": response,
            "asr_status": None,
        }

    def handle_audio(self, sample_rate: int, data) -> dict:
        """Transcribe audio, then classify + respond -- unless the ASR gate rejected the clip."""
        if self.recognizer is None:
            raise RuntimeError("SpeechRecognizer was not loaded (load_asr=False); cannot handle audio.")

        asr_result = self.recognizer.transcribe(sample_rate, data)
        status = asr_result["status"]

        if status not in _ASR_STATUSES_TO_CLASSIFY:
            return {
                "source": "voice",
                "transcript": asr_result["text"],
                "intent": "oos",
                "confidence": 0.0,
                "top3": [],
                "is_fallback": True,
                "fallback_reason": status,
                "response": ASR_STATUS_MESSAGES.get(status, ASR_STATUS_MESSAGES["error"]),
                "asr_status": status,
            }

        prediction = self.predictor.predict(asr_result["text"])
        response = self.responder.respond(prediction["intent"])

        return {
            "source": "voice",
            "transcript": asr_result["text"],
            "intent": prediction["intent"],
            "confidence": prediction["confidence"],
            "top3": prediction["top3"],
            "is_fallback": prediction["is_fallback"],
            "fallback_reason": prediction["fallback_reason"],
            "response": response,
            "asr_status": status,
        }
