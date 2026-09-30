"""
Pure text-formatting helpers shared by both frontends (src/ui.py for
Gradio, streamlit_app.py for Streamlit).

Every function here takes plain data (a pipeline result dict, a label
string, a list of (label, prob) tuples) and returns plain data (a string
or a dict) -- no Gradio/Streamlit imports, no model calls, so both UIs
render identical wording and this module can be unit tested without
loading anything.
"""

import json
from pathlib import Path

# Example queries shown as clickable buttons in both frontends (5 in-scope,
# 1 out-of-scope).
TEXT_EXAMPLES = [
    "book me a flight to chicago",
    "what's my account balance",
    "set an alarm for 7 am",
    "tell me a joke",
    "what's the weather like today",
    "who won the 1998 football world cup",  # out-of-scope example
]

# Understanding-panel text for ASR statuses that reject a clip before it
# ever reaches the classifier (see src.asr.SpeechRecognizer.transcribe).
# Distinct from src.responses.ASR_STATUS_MESSAGES, which is the *bot
# response* shown for the same statuses -- this is the shorter reason
# shown in the "Understanding" panel.
ASR_REJECTION_UNDERSTANDING = {
    "too_short": "Audio rejected — the clip was too short to transcribe.",
    "silent": "Audio rejected — the clip appears to be silence.",
    "no_speech": "No speech detected in that clip.",
    "error": "Something went wrong while processing that audio.",
}


def format_intent_name(label: str) -> str:
    """"book_flight" -> "Book Flight"."""
    return label.replace("_", " ").title()


def top3_dict(top3: list[tuple[str, float]]) -> dict[str, float]:
    """[("book_flight", 0.97), ...] -> {"Book Flight": 0.97, ...}."""
    return {format_intent_name(label): prob for label, prob in top3}


def format_understanding(result: dict, threshold: float) -> str:
    """
    Build the "Understanding" text for one VoiceChatbot result dict.

    Never reports "oos" alongside a confidence number that belongs to a
    different intent: the predicted-oos case shows no number at all, and
    the low-confidence case shows the *actual* top guess and its own
    confidence, not the (meaningless) "oos" label.
    """
    asr_status = result.get("asr_status")
    if asr_status is not None and asr_status not in ("ok", "truncated"):
        return ASR_REJECTION_UNDERSTANDING.get(asr_status, "Audio was rejected before transcription.")

    reason = result["fallback_reason"]

    if reason is None:
        pretty = format_intent_name(result["intent"])
        return f"Intent: {pretty} — {result['confidence'] * 100:.1f}% confidence"

    if reason == "predicted_oos":
        return "Out of scope (the model predicted the out-of-scope class)"

    if reason == "low_confidence":
        top_label, top_conf = result["top3"][0]
        pretty = format_intent_name(top_label)
        return (
            f"Low confidence — top guess: {pretty} ({top_conf * 100:.0f}%), "
            f"below the {threshold * 100:.0f}% threshold"
        )

    if reason == "empty_text":
        return "No text detected — please say or type something."

    return "Unable to determine intent."


def load_about_markdown(artifacts_dir) -> str:
    """
    Build the "About this model" text from the saved training artifacts
    (config.json, metrics.json). Shared by both frontends so the "About"
    section reads identically in Gradio and Streamlit.
    """
    artifacts_dir = Path(artifacts_dir)

    with (artifacts_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    with (artifacts_dir / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    threshold = config["threshold"]
    scored = metrics["with_threshold"]

    return f"""\
**Pipeline:** microphone or text → Whisper `base.en` (speech-to-text) → BiLSTM intent classifier → response template.

**Dataset:** CLINC150 "plus" — 150 everyday intents across 10 domains (banking, travel, utility, small talk, etc.) plus an out-of-scope ("oos") class.

**Test set results (threshold applied):**
- In-scope accuracy: {scored['in_scope_accuracy'] * 100:.1f}%
- Out-of-scope recall: {scored['oos_recall'] * 100:.1f}%
- Macro F1: {scored['macro_f1'] * 100:.1f}%

**Confidence threshold:** {threshold:.2f} — tuned to maximize *balanced accuracy* (the average of in-scope accuracy and OOS recall) rather than plain accuracy, since plain accuracy would favor ignoring out-of-scope detection on this dataset.
"""
