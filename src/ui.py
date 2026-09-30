"""
Gradio UI: handler functions + build_demo(bot).

Handlers (process_text, process_voice) are plain functions that take the
VoiceChatbot instance as an explicit argument, so they can be unit tested
against a stub bot without touching Gradio or loading any models. app.py
stays thin: it only constructs the one real VoiceChatbot, warms it up, and
calls build_demo(bot).
"""

import traceback

import gradio as gr

from src.config import ARTIFACTS_DIR
from src.formatting import TEXT_EXAMPLES, format_understanding, load_about_markdown, top3_dict

# Friendly messages shown instead of letting an exception or empty input
# reach the user as a Gradio error.
_EMPTY_TEXT_MESSAGE = "Please type a message first."
_EMPTY_AUDIO_MESSAGE = "Please record or upload some audio first."
_ERROR_MESSAGE = "Something went wrong on my end. Please try again."


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

    understanding = format_understanding(result, bot.predictor.threshold)
    top3 = top3_dict(result["top3"])
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

    understanding = format_understanding(result, bot.predictor.threshold)
    top3 = top3_dict(result["top3"])
    transcript = result["transcript"]
    user_turn = f"🎤 {transcript}" if transcript else "🎤 (no speech detected)"
    history = history + [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": result["response"]},
    ]
    return transcript, understanding, top3, result["response"], history


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
            gr.Markdown(load_about_markdown(ARTIFACTS_DIR))

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
