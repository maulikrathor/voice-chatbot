"""
Gradio UI: handler functions + build_demo(bot).

Handlers (process_text, process_voice) are plain functions that take the
VoiceChatbot instance as an explicit argument, so they can be unit tested
against a stub bot without touching Gradio or loading any models. app.py
stays thin: it only constructs the one real VoiceChatbot, warms it up, and
calls build_demo(bot).
"""

import json
import traceback

import gradio as gr

from src.config import ARTIFACTS_DIR, CONFIG_FILENAME, METRICS_FILENAME

# Friendly messages shown instead of letting an exception or empty input
# reach the user as a Gradio error.
_EMPTY_TEXT_MESSAGE = "Please type a message first."
_EMPTY_AUDIO_MESSAGE = "Please record or upload some audio first."
_ERROR_MESSAGE = "Something went wrong on my end. Please try again."

# Understanding-panel text for ASR statuses that reject a clip before it
# ever reaches the classifier (see src.asr.SpeechRecognizer.transcribe).
_ASR_REJECTION_UNDERSTANDING = {
    "too_short": "Audio rejected — the clip was too short to transcribe.",
    "silent": "Audio rejected — the clip appears to be silence.",
    "no_speech": "No speech detected in that clip.",
    "error": "Something went wrong while processing that audio.",
}

TEXT_EXAMPLES = [
    "book me a flight to chicago",
    "what's my account balance",
    "set an alarm for 7 am",
    "tell me a joke",
    "what's the weather like today",
    "who won the 1998 football world cup",  # out-of-scope example
]


def _format_intent_name(label: str) -> str:
    """"book_flight" -> "Book Flight"."""
    return label.replace("_", " ").title()


def _top3_dict(top3: list[tuple[str, float]]) -> dict[str, float]:
    """[("book_flight", 0.97), ...] -> {"Book Flight": 0.97, ...} for gr.Label."""
    return {_format_intent_name(label): prob for label, prob in top3}


def _format_understanding(result: dict, threshold: float) -> str:
    """
    Build the "Understanding" markdown for one pipeline result.

    Never reports "oos" alongside a confidence number that belongs to a
    different intent: the predicted-oos case shows no number at all, and
    the low-confidence case shows the *actual* top guess and its own
    confidence, not the (meaningless) "oos" label.
    """
    asr_status = result.get("asr_status")
    if asr_status is not None and asr_status not in ("ok", "truncated"):
        return _ASR_REJECTION_UNDERSTANDING.get(asr_status, "Audio was rejected before transcription.")

    reason = result["fallback_reason"]

    if reason is None:
        pretty = _format_intent_name(result["intent"])
        return f"Intent: {pretty} — {result['confidence'] * 100:.1f}% confidence"

    if reason == "predicted_oos":
        return "Out of scope (the model predicted the out-of-scope class)"

    if reason == "low_confidence":
        top_label, top_conf = result["top3"][0]
        pretty = _format_intent_name(top_label)
        return (
            f"Low confidence — top guess: {pretty} ({top_conf * 100:.0f}%), "
            f"below the {threshold * 100:.0f}% threshold"
        )

    if reason == "empty_text":
        return "No text detected — please say or type something."

    return "Unable to determine intent."


def process_text(bot, text: str, history: list[dict] | None) -> tuple[str, str, dict, str, list[dict]]:
    """
    Classify typed text and generate a response.

    Returns (you_said, understanding_md, top3_label_dict, bot_response, history).
    """
    history = list(history or [])

    if not text or not text.strip():
        return "", _EMPTY_TEXT_MESSAGE, {}, "", history

    try:
        result = bot.handle_text(text)
    except Exception:
        traceback.print_exc()
        return text, _ERROR_MESSAGE, {}, "", history

    understanding = _format_understanding(result, bot.predictor.threshold)
    top3 = _top3_dict(result["top3"])
    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": result["response"]},
    ]
    return result["transcript"], understanding, top3, result["response"], history


def process_voice(bot, audio, history: list[dict] | None) -> tuple[str, str, dict, str, list[dict]]:
    """
    Transcribe + classify recorded/uploaded audio and generate a response.

    Returns (you_said, understanding_md, top3_label_dict, bot_response, history).
    """
    history = list(history or [])

    if audio is None:
        return "", _EMPTY_AUDIO_MESSAGE, {}, "", history

    try:
        sample_rate, data = audio
        result = bot.handle_audio(sample_rate, data)
    except Exception:
        traceback.print_exc()
        return "", _ERROR_MESSAGE, {}, "", history

    understanding = _format_understanding(result, bot.predictor.threshold)
    top3 = _top3_dict(result["top3"])
    transcript = result["transcript"]
    user_turn = f"🎤 {transcript}" if transcript else "🎤 (no speech detected)"
    history = history + [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": result["response"]},
    ]
    return transcript, understanding, top3, result["response"], history


def _load_about_markdown() -> str:
    """Build the "About this model" accordion text from the saved training artifacts."""
    with (ARTIFACTS_DIR / CONFIG_FILENAME).open("r", encoding="utf-8") as f:
        config = json.load(f)
    with (ARTIFACTS_DIR / METRICS_FILENAME).open("r", encoding="utf-8") as f:
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


def build_demo(bot) -> gr.Blocks:
    """Construct the Gradio Blocks UI wired to the given VoiceChatbot (or stub)."""
    with gr.Blocks(title="Voice Intent Chatbot") as demo:
        gr.Markdown("# Voice Intent Chatbot")
        gr.Markdown(
            "Speak into your microphone (or upload/type) and the bot will transcribe, "
            "classify, and respond. It recognizes 150 everyday intents across 10 domains "
            "(banking, travel, small talk, utility, and more)."
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak or upload audio")
                voice_btn = gr.Button("Send voice")
            with gr.Column():
                text_input = gr.Textbox(label="Or type a message", placeholder="e.g. book me a flight to chicago")
                text_btn = gr.Button("Send text")
                gr.Examples(examples=TEXT_EXAMPLES, inputs=text_input, label="Try one of these")

        you_said = gr.Textbox(label="You said", interactive=False)
        understanding = gr.Markdown(label="Understanding", show_label=True, container=True)
        top3_label = gr.Label(label="Top-3 intents", num_top_classes=3)
        bot_response = gr.Textbox(label="Bot response", interactive=False)
        # Gradio 6.27's gr.Chatbot always uses the messages format
        # ({"role", "content"} dicts) -- the `type="messages"` kwarg from
        # older Gradio versions no longer exists and raises a TypeError.
        chatbot = gr.Chatbot(label="Conversation")
        clear_btn = gr.Button("Clear")

        with gr.Accordion("About this model", open=False):
            gr.Markdown(_load_about_markdown())

        outputs = [you_said, understanding, top3_label, bot_response, chatbot]

        voice_btn.click(fn=lambda audio, history: process_voice(bot, audio, history), inputs=[audio_input, chatbot], outputs=outputs)
        # Auto-run on stop_recording (mic only; fires when the user stops
        # recording) in addition to the explicit button, per the spec.
        audio_input.stop_recording(fn=lambda audio, history: process_voice(bot, audio, history), inputs=[audio_input, chatbot], outputs=outputs)

        text_btn.click(fn=lambda text, history: process_text(bot, text, history), inputs=[text_input, chatbot], outputs=outputs)
        text_input.submit(fn=lambda text, history: process_text(bot, text, history), inputs=[text_input, chatbot], outputs=outputs)

        def _clear():
            return None, "", "", "", {}, "", []

        clear_btn.click(
            fn=_clear,
            inputs=None,
            outputs=[audio_input, text_input, you_said, understanding, top3_label, bot_response, chatbot],
        )

    return demo
