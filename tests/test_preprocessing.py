"""
Tests for src.text_preprocessing: normalize_text, tokenize, and Vocab.
"""

import json

from src.text_preprocessing import Vocab, normalize_text, tokenize


def test_normalize_text_handles_punctuation_case_and_spacing():
    assert normalize_text("What's the  WEATHER?!") == "what's the weather"


def test_normalize_text_strips_leading_and_trailing_whitespace():
    assert normalize_text("  hello world  ") == "hello world"


def test_normalize_text_removes_disallowed_characters():
    assert normalize_text("Call me @ 5:00pm, please!") == "call me 500pm please"


def test_tokenize_splits_normalized_text_on_whitespace():
    assert tokenize("What's the  WEATHER?!") == ["what's", "the", "weather"]


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("   !!! ???  ") == []


def test_vocab_reserves_pad_and_unk_ids():
    vocab = Vocab.build(["hello world", "hello there"], min_freq=1)
    assert vocab.token_to_id["<pad>"] == 0
    assert vocab.token_to_id["<unk>"] == 1


def test_vocab_min_freq_drops_rare_tokens():
    texts = ["a a a b b c"]  # a:3, b:2, c:1
    vocab = Vocab.build(texts, min_freq=2)
    assert "a" in vocab.token_to_id
    assert "b" in vocab.token_to_id
    assert "c" not in vocab.token_to_id


def test_vocab_encode_pads_short_sequences():
    vocab = Vocab.build(["hello world"], min_freq=1)
    ids = vocab.encode("hello", max_len=5)
    assert len(ids) == 5
    assert ids[0] == vocab.token_to_id["hello"]
    assert ids[1:] == [0, 0, 0, 0]


def test_vocab_encode_truncates_long_sequences():
    vocab = Vocab.build(["one two three four five"], min_freq=1)
    ids = vocab.encode("one two three four five", max_len=3)
    assert len(ids) == 3
    assert ids == [
        vocab.token_to_id["one"],
        vocab.token_to_id["two"],
        vocab.token_to_id["three"],
    ]


def test_vocab_encode_maps_unseen_words_to_unk():
    vocab = Vocab.build(["hello world"], min_freq=1)
    ids = vocab.encode("goodbye", max_len=1)
    assert ids == [vocab.token_to_id["<unk>"]]


def test_vocab_save_load_round_trip(tmp_path):
    vocab = Vocab.build(["hello world", "goodbye world"], min_freq=1)
    path = tmp_path / "vocab.json"
    vocab.save(path)

    # Confirm it was written as plain JSON, as save()/load() promise.
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw == vocab.token_to_id

    loaded = Vocab.load(path)
    assert loaded.token_to_id == vocab.token_to_id
    assert len(loaded) == len(vocab)
