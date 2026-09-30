"""
Tests for src.ui handlers (process_text, process_voice, build_demo).

process_text tests use a real VoiceChatbot(load_asr=False) -- fast, since
that skips loading Whisper entirely. process_voice tests use small stub
bots instead of any real model, since they only need to exercise
process_voice's own control flow (empty input / exceptions / ASR
rejection statuses), not real audio transcription.

These tests check that the handlers wire bot output through to
src.formatting correctly (by comparing against format_understanding's own
output, not a duplicated literal string) and manage history/empty-input/
exception behavior. The exact wording of each formatting case is covered
once, in tests/test_formatting.py.

Handlers and build_demo live in src/ui.py (not app.py), so importing this
module -- or app.py itself -- is never required to load any models here.
"""

from src.formatting import format_understanding, top3_dict
from src.pipeline import VoiceChatbot
from src.responses import ASR_STATUS_MESSAGES
from src.ui import build_demo, process_text, process_voice


# --- process_text --------------------------------------------------------


def test_process_text_returns_all_outputs_and_grows_history():
    bot = VoiceChatbot(load_asr=False)

    you_said, understanding, top3, response, history = process_text(
        bot, "what's the weather like today", []
    )

    assert you_said == "what's the weather like today"
    assert understanding.startswith("Intent:")
    assert isinstance(top3, dict) and top3
    assert isinstance(response, str) and response.strip()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "what's the weather like today"}
    assert history[1]["role"] == "assistant" and history[1]["content"] == response


def test_process_text_empty_input_gives_friendly_message_and_leaves_history():
    bot = VoiceChatbot(load_asr=False)

    you_said, understanding, top3, response, history = process_text(bot, "   ", [])

    assert you_said == ""
    assert understanding == "Please type a message first."
    assert top3 == {}
    assert response == ""
    assert history == []


class _StubPredictor:
    threshold = 0.5


class _LowConfidenceBot:
    """Fake bot whose handle_text always returns a canned low-confidence result."""

    predictor = _StubPredictor()

    def handle_text(self, text):
        return {
            "source": "text",
            "transcript": text,
            "intent": "oos",
            "confidence": 0.42,
            "top3": [("book_hotel", 0.48), ("book_flight", 0.30), ("oos", 0.10)],
            "is_fallback": True,
            "fallback_reason": "low_confidence",
            "response": "I'm not sure I understood that.",
            "asr_status": None,
        }


def test_process_text_uses_shared_formatting_helpers():
    bot = _LowConfidenceBot()
    canned_result = bot.handle_text("some ambiguous query")

    _, understanding, top3, response, history = process_text(bot, "some ambiguous query", [])

    # process_text's outputs must match calling the shared helpers directly
    # on the same bot result -- i.e. it isn't reformatting things its own way.
    assert understanding == format_understanding(canned_result, bot.predictor.threshold)
    assert top3 == top3_dict(canned_result["top3"])
    assert len(history) == 2


# --- process_voice ---------------------------------------------------------


class _RaisingBot:
    predictor = _StubPredictor()

    def handle_audio(self, sample_rate, data):
        raise RuntimeError("boom")


def test_process_voice_none_audio_gives_friendly_message():
    you_said, understanding, top3, response, history = process_voice(_RaisingBot(), None, [])

    assert you_said == ""
    assert understanding == "Please record or upload some audio first."
    assert top3 == {}
    assert response == ""
    assert history == []


def test_process_voice_bot_exception_gives_friendly_message_not_raise():
    fake_audio = (16000, [0.0] * 16000)

    you_said, understanding, top3, response, history = process_voice(
        _RaisingBot(), fake_audio, []
    )

    assert understanding == "Something went wrong on my end. Please try again."
    assert top3 == {}
    assert history == []


class _SilentAsrBot:
    """Fake bot whose handle_audio always returns an ASR "silent" rejection."""

    predictor = _StubPredictor()

    def handle_audio(self, sample_rate, data):
        return {
            "source": "voice",
            "transcript": "",
            "intent": "oos",
            "confidence": 0.0,
            "top3": [],
            "is_fallback": True,
            "fallback_reason": "silent",
            "response": ASR_STATUS_MESSAGES["silent"],
            "asr_status": "silent",
        }


def test_process_voice_asr_rejection_uses_shared_formatting_and_marks_history():
    bot = _SilentAsrBot()
    fake_audio = (16000, [0.0] * 16000)
    canned_result = bot.handle_audio(*fake_audio)

    you_said, understanding, top3, response, history = process_voice(bot, fake_audio, [])

    assert you_said == ""
    assert understanding == format_understanding(canned_result, bot.predictor.threshold)
    assert top3 == {}
    assert response == ASR_STATUS_MESSAGES["silent"]
    assert len(history) == 2
    assert history[0]["content"] == "🎤 (no speech detected)"


# --- build_demo --------------------------------------------------------------


def test_build_demo_constructs_without_error_with_a_stub_bot():
    demo = build_demo(_RaisingBot())
    assert demo is not None
