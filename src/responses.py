"""
Intent-to-response mapping.

ResponseEngine loads data/responses.json (exactly one entry per label in
labels.json) and turns a predicted intent into a response string. Four
labels get a *dynamic* handler instead of a static template -- "time" and
"date" (Asia/Kolkata) and, in addition to the ARCHITECTURE.md-specified
pair, "flip_coin" and "roll_dice" -- because their whole point is to
return a fresh, real value on every call rather than a fixed sentence.

System messages for ASR-side statuses (too_short, silent, no_speech,
error) live here as constants, not in responses.json, because they are
not tied to any predicted label -- they fire instead of running the
classifier at all (see src/pipeline.py).
"""

import json
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import ARTIFACTS_DIR, DATA_DIR, LABELS_FILENAME

RESPONSES_FILENAME = "responses.json"

# CLINC150 "plus" has no US/India distinction; Asia/Kolkata is fixed by
# ARCHITECTURE.md section 4 for the time/date dynamic handlers.
TIMEZONE = ZoneInfo("Asia/Kolkata")

# Labels that must exist in labels.json for their dynamic handler to be
# wired up. Checked at construction time; any that are missing are simply
# skipped (and reported), so a labels.json change can't crash the engine.
_DYNAMIC_HANDLER_LABELS = ("time", "date", "flip_coin", "roll_dice")

# System messages for ASR outcomes that never reach the classifier.
# Keyed by src.asr.SpeechRecognizer status strings.
ASR_STATUS_MESSAGES = {
    "too_short": "I didn't catch that — the clip was too short. Please try again with a slightly longer recording.",
    "silent": "I didn't catch that — the audio seemed to be silence. Please try again and speak clearly into the microphone.",
    "no_speech": "I didn't catch any speech in that clip. Could you try again?",
    "error": "Something went wrong while processing that audio. Please try again.",
}


class ResponseEngine:
    """Maps a predicted intent to a response string, with dynamic handlers and a random.Random for testability."""

    def __init__(self, data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR, rng: random.Random | None = None):
        data_dir = Path(data_dir)
        artifacts_dir = Path(artifacts_dir)
        self.rng = rng if rng is not None else random.Random()

        with (artifacts_dir / LABELS_FILENAME).open("r", encoding="utf-8") as f:
            labels: list[str] = json.load(f)

        with (data_dir / RESPONSES_FILENAME).open("r", encoding="utf-8") as f:
            self.responses: dict[str, list[str]] = json.load(f)

        self._validate_labels_match(labels)

        # Only wire up handlers for labels that actually exist in this
        # label set; report (rather than crash on) any that don't.
        self.enabled_dynamic_labels = [
            label for label in _DYNAMIC_HANDLER_LABELS if label in self.responses
        ]
        self.skipped_dynamic_labels = [
            label for label in _DYNAMIC_HANDLER_LABELS if label not in self.responses
        ]

        self._dynamic_handlers = {
            "time": self._handle_time,
            "date": self._handle_date,
            "flip_coin": self._handle_flip_coin,
            "roll_dice": self._handle_roll_dice,
        }

    def _validate_labels_match(self, labels: list[str]) -> None:
        """Ensure responses.json has exactly one entry per label, no more, no fewer."""
        label_set = set(labels)
        response_keys = set(self.responses.keys())

        missing = sorted(label_set - response_keys)
        extra = sorted(response_keys - label_set)
        if missing or extra:
            raise ValueError(
                "data/responses.json keys do not match artifacts/labels.json exactly. "
                f"Missing labels (in labels.json, not in responses.json): {missing}. "
                f"Extra labels (in responses.json, not in labels.json): {extra}."
            )

    def respond(self, intent: str) -> str:
        """
        Return a response string for a predicted intent.

        Dynamic handlers (time/date/flip_coin/roll_dice) take precedence
        over their static template when wired up. Unknown intents (not in
        responses.json) fall back to the "oos" response.
        """
        if intent in self.enabled_dynamic_labels:
            return self._dynamic_handlers[intent]()

        variants = self.responses.get(intent) or self.responses["oos"]
        return self.rng.choice(variants)

    def _handle_time(self) -> str:
        now = datetime.now(TIMEZONE)
        return f"It's currently {now.strftime('%I:%M %p')} (Asia/Kolkata)."

    def _handle_date(self) -> str:
        now = datetime.now(TIMEZONE)
        return f"Today's date is {now.strftime('%A, %B %d, %Y')} (Asia/Kolkata)."

    def _handle_flip_coin(self) -> str:
        result = self.rng.choice(["heads", "tails"])
        return f"I flipped a coin: {result}!"

    def _handle_roll_dice(self) -> str:
        result = self.rng.randint(1, 6)
        return f"I rolled the dice: {result}!"
