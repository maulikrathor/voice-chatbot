"""
Gradio UI for the voice chatbot.

Thin entry point: builds the one real VoiceChatbot (loading the BiLSTM and
Whisper models once), warms up Whisper, builds the Gradio Blocks demo from
src.ui, and launches it. All handler logic and UI construction lives in
src/ui.py so it can be unit-tested without loading any models (see
tests/test_app.py).
"""

import time

import numpy as np

from src.config import TARGET_SAMPLE_RATE
from src.pipeline import VoiceChatbot
from src.ui import build_demo


def _warm_up(bot: VoiceChatbot) -> None:
    """
    Run one Whisper call before serving real requests, so the first user
    doesn't pay the model's one-time graph/kernel warm-up cost.

    Uses ~1 s of low-amplitude noise and calls the ASR pipeline directly
    (bypassing SpeechRecognizer.transcribe's silence gate, which would
    otherwise reject this clip and skip the model entirely).
    """
    noise = (np.random.default_rng(0).standard_normal(TARGET_SAMPLE_RATE) * 0.001).astype(np.float32)
    bot.recognizer._asr_pipeline({"raw": noise, "sampling_rate": TARGET_SAMPLE_RATE})


print("Loading models...")
_load_start = time.perf_counter()
chatbot = VoiceChatbot(load_asr=True)
_load_elapsed = time.perf_counter() - _load_start
print(f"Model load time: {_load_elapsed:.2f}s")

print("Warming up Whisper...")
_warmup_start = time.perf_counter()
_warm_up(chatbot)
_warmup_elapsed = time.perf_counter() - _warmup_start
print(f"Warm-up time: {_warmup_elapsed:.2f}s")

demo = build_demo(chatbot)

if __name__ == "__main__":
    demo.launch()
