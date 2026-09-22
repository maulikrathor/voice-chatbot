"""
Tests for src.responses.ResponseEngine: label coverage, variant sanity,
dynamic handlers, unknown-intent fallback, and deterministic output with a
seeded RNG.
"""

import json
import random

import pytest

from src.config import ARTIFACTS_DIR, DATA_DIR, LABELS_FILENAME
from src.responses import ResponseEngine


def _load_labels():
    with (ARTIFACTS_DIR / LABELS_FILENAME).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_responses_json():
    with (DATA_DIR / "responses.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def test_full_label_coverage():
    labels = set(_load_labels())
    response_keys = set(_load_responses_json().keys())
    assert response_keys == labels


def test_no_empty_variants():
    responses = _load_responses_json()
    for label, variants in responses.items():
        assert isinstance(variants, list) and len(variants) >= 1, label
        for variant in variants:
            assert isinstance(variant, str) and variant.strip(), (label, variant)


def test_engine_loads_successfully():
    engine = ResponseEngine(rng=random.Random(0))
    assert len(engine.responses) == len(_load_labels())


def test_dynamic_handlers_return_non_empty_strings():
    engine = ResponseEngine(rng=random.Random(0))

    # All four dynamic-handler labels are expected to exist in this
    # project's labels.json (verified against artifacts/labels.json).
    assert engine.skipped_dynamic_labels == []
    assert set(engine.enabled_dynamic_labels) == {"time", "date", "flip_coin", "roll_dice"}

    for label in ("time", "date", "flip_coin", "roll_dice"):
        response = engine.respond(label)
        assert isinstance(response, str) and response.strip()


def test_unknown_intent_returns_oos_response():
    engine = ResponseEngine(rng=random.Random(0))
    oos_variants = set(engine.responses["oos"])

    response = engine.respond("not_a_real_intent_xyz")
    assert response in oos_variants


def test_respond_is_deterministic_with_seeded_rng():
    engine_a = ResponseEngine(rng=random.Random(42))
    engine_b = ResponseEngine(rng=random.Random(42))

    # Same seed, same sequence of calls -> identical sequence of responses.
    sequence_a = [engine_a.respond("greeting") for _ in range(5)]
    sequence_b = [engine_b.respond("greeting") for _ in range(5)]
    assert sequence_a == sequence_b


def test_validate_labels_match_raises_on_mismatch(tmp_path):
    # Build a responses.json missing one label and adding a bogus one, and
    # confirm the engine refuses to load it with a clear error.
    labels = _load_labels()
    bad_responses = {label: ["placeholder response."] for label in labels[1:]}
    bad_responses["not_a_real_label"] = ["placeholder response."]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with (data_dir / "responses.json").open("w", encoding="utf-8") as f:
        json.dump(bad_responses, f)

    with pytest.raises(ValueError, match="do not match"):
        ResponseEngine(data_dir=data_dir, artifacts_dir=ARTIFACTS_DIR, rng=random.Random(0))
