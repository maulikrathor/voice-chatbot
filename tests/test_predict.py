"""
Tests for src.predict.IntentPredictor, using the real trained artifacts
in artifacts/ (produced by `python -m src.train`).
"""

from src.predict import IntentPredictor

# (query, expected_label) -- all expected labels verified to exist in
# artifacts/labels.json. Chosen to be unambiguous, everyday phrasings of
# in-domain CLINC150 intents.
QUERIES = [
    ("what's the weather like today", "weather"),
    ("what time is it right now", "time"),
    ("tell me a joke", "tell_joke"),
    ("i want to book a flight to paris", "book_flight"),
    ("can you transfer 100 dollars to my savings account", "transfer"),
    ("hello there", "greeting"),
    ("set an alarm for 7am", "alarm"),
    ("translate hello into spanish", "translate"),
    ("what is the capital of france", "oos"),
    ("goodbye", "goodbye"),
]


def test_ten_in_domain_queries_at_least_eight_correct():
    predictor = IntentPredictor()

    rows = []
    correct = 0
    for query, expected in QUERIES:
        result = predictor.predict(query)
        is_correct = result["intent"] == expected
        correct += is_correct
        rows.append((query, expected, result["intent"], result["confidence"], is_correct))

    print("\nquery | expected | predicted | confidence | correct")
    for query, expected, predicted, confidence, is_correct in rows:
        print(f"  {query!r:55} {expected:15} {predicted:15} {confidence:.3f}   {is_correct}")

    assert correct >= 8, f"Only {correct}/10 queries predicted correctly."


def test_empty_text_gives_empty_text_fallback():
    predictor = IntentPredictor()

    for text in ("", "   ", "??? !!!"):
        result = predictor.predict(text)
        assert result["is_fallback"] is True
        assert result["fallback_reason"] == "empty_text"
        assert result["intent"] == "oos"
        assert result["confidence"] == 0.0


def test_result_dict_has_all_expected_keys():
    predictor = IntentPredictor()
    result = predictor.predict("what's the weather like today")

    expected_keys = {
        "input_text",
        "normalized_text",
        "intent",
        "confidence",
        "top3",
        "is_fallback",
        "fallback_reason",
    }
    assert set(result.keys()) == expected_keys


def test_confidence_is_in_valid_range():
    predictor = IntentPredictor()
    for query, _expected in QUERIES:
        result = predictor.predict(query)
        assert 0.0 <= result["confidence"] <= 1.0
