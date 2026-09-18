"""
Tests for src.data.load_clinc: split sizes, label count, and label id
range, for the CLINC150 "plus" dataset.
"""

from src.data import load_clinc

# Loading the dataset hits the network (or the local HF cache); load it once
# for all tests in this module rather than once per test.
_TRAIN, _VAL, _TEST, _ID2LABEL, _LABEL2ID = load_clinc()


def test_split_sizes():
    assert len(_TRAIN) == 15250
    assert len(_VAL) == 3100
    assert len(_TEST) == 5500


def test_label_count_includes_oos():
    assert len(_ID2LABEL) == 151
    assert "oos" in _ID2LABEL.values()
    assert "oos" in _LABEL2ID


def test_label_ids_in_range():
    num_labels = len(_ID2LABEL)
    for split in (_TRAIN, _VAL, _TEST):
        for _, label_id in split:
            assert 0 <= label_id < num_labels


def test_label_maps_are_consistent():
    for label_id, label_name in _ID2LABEL.items():
        assert _LABEL2ID[label_name] == label_id
