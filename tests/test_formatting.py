"""
Tests for the pure formatting helpers in src.formatting, shared by both
frontends (src/ui.py for Gradio, streamlit_app.py for Streamlit).

These are plain-data unit tests: no bot, no models, no UI framework.
"""

from src.formatting import (
    ASR_REJECTION_UNDERSTANDING,
    format_intent_name,
    format_understanding,
    load_about_markdown,
    top3_dict,
)
from src.config import ARTIFACTS_DIR


def _make_result(**overrides) -> dict:
    base = {
        "transcript": "some text",
        "intent": "book_flight",
        "confidence": 0.966,
        "top3": [("book_flight", 0.966), ("book_hotel", 0.02), ("car_rental", 0.01)],
        "is_fallback": False,
        "fallback_reason": None,
        "asr_status": None,
    }
    base.update(overrides)
    return base


def test_format_intent_name_replaces_underscores_and_title_cases():
    assert format_intent_name("book_flight") == "Book Flight"
    assert format_intent_name("oos") == "Oos"
    assert format_intent_name("what_can_i_ask_you") == "What Can I Ask You"


def test_top3_dict_converts_label_list_to_pretty_dict():
    top3 = [("book_flight", 0.9), ("book_hotel", 0.05), ("oos", 0.05)]
    assert top3_dict(top3) == {"Book Flight": 0.9, "Book Hotel": 0.05, "Oos": 0.05}


def test_format_understanding_normal_case():
    result = _make_result()
    assert format_understanding(result, threshold=0.5) == "Intent: Book Flight — 96.6% confidence"


def test_format_understanding_predicted_oos_never_shows_a_number():
    result = _make_result(
        intent="oos",
        confidence=0.79,
        top3=[("oos", 0.79), ("smart_home", 0.10), ("directions", 0.05)],
        is_fallback=True,
        fallback_reason="predicted_oos",
    )
    understanding = format_understanding(result, threshold=0.5)
    assert understanding == "Out of scope (the model predicted the out-of-scope class)"
    # The whole point of this fallback: never pair "oos" with a confidence number.
    assert "%" not in understanding


def test_format_understanding_low_confidence_reports_the_actual_top_guess():
    result = _make_result(
        intent="oos",
        confidence=0.42,
        top3=[("book_hotel", 0.48), ("book_flight", 0.30), ("oos", 0.10)],
        is_fallback=True,
        fallback_reason="low_confidence",
    )
    understanding = format_understanding(result, threshold=0.5)
    assert understanding == "Low confidence — top guess: Book Hotel (48%), below the 50% threshold"
    # Must never show "oos" as the reported top guess here -- it's not the real top-1.
    assert "Oos" not in understanding


def test_format_understanding_empty_text():
    result = _make_result(
        intent="oos", confidence=0.0, top3=[], is_fallback=True, fallback_reason="empty_text"
    )
    assert format_understanding(result, threshold=0.5) == "No text detected — please say or type something."


def test_format_understanding_asr_rejection_states_the_reason():
    for status, expected in ASR_REJECTION_UNDERSTANDING.items():
        result = _make_result(
            transcript="",
            intent="oos",
            confidence=0.0,
            top3=[],
            is_fallback=True,
            fallback_reason=status,
            asr_status=status,
        )
        assert format_understanding(result, threshold=0.5) == expected


def test_format_understanding_asr_ok_status_is_not_treated_as_a_rejection():
    # "ok"/"truncated" mean the clip WAS transcribed -- must fall through
    # to normal intent formatting, not an ASR-rejection message.
    result = _make_result(asr_status="ok")
    assert format_understanding(result, threshold=0.5) == "Intent: Book Flight — 96.6% confidence"


def test_load_about_markdown_mentions_the_pipeline_and_threshold():
    markdown = load_about_markdown(ARTIFACTS_DIR)
    assert "Whisper" in markdown
    assert "BiLSTM" in markdown
    assert "CLINC150" in markdown
