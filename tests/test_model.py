"""
Tests for src.model.BiLSTMClassifier: output shape, padding invariance,
robustness to empty input, state_dict save/load, and an overfit sanity
check. Uses tiny synthetic tensors only -- no dataset download needed.
"""

import torch
from torch import nn

from src.model import BiLSTMClassifier


def _make_model(vocab_size=20, num_classes=5, pad_idx=0, dropout=0.0):
    torch.manual_seed(0)
    return BiLSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=8,
        hidden_dim=8,
        dropout=dropout,
        pad_idx=pad_idx,
    )


def test_forward_output_shape():
    model = _make_model(vocab_size=20, num_classes=5)
    model.eval()

    batch_size, max_len = 4, 10
    token_ids = torch.randint(1, 20, (batch_size, max_len))
    lengths = torch.tensor([10, 7, 3, 1])

    logits = model(token_ids, lengths)
    assert logits.shape == (batch_size, 5)


def test_padding_invariance():
    """The same real tokens padded to different total lengths must give identical logits."""
    model = _make_model(vocab_size=20, num_classes=5)
    model.eval()

    tokens = [3, 7, 9, 2]
    length = torch.tensor([len(tokens)])

    short = torch.tensor([tokens + [0] * (6 - len(tokens))])
    long = torch.tensor([tokens + [0] * (16 - len(tokens))])

    logits_short = model(short, length)
    logits_long = model(long, length)

    assert torch.allclose(logits_short, logits_long, atol=1e-6)


def test_empty_sequence_produces_finite_logits():
    """An all-pad row (length 0, clamped internally) must not produce NaN/inf."""
    model = _make_model(vocab_size=20, num_classes=5)
    model.eval()

    token_ids = torch.zeros((1, 8), dtype=torch.long)
    lengths = torch.tensor([0])

    logits = model(token_ids, lengths)
    assert torch.isfinite(logits).all()


def test_state_dict_save_load_round_trip(tmp_path):
    model = _make_model(vocab_size=20, num_classes=5)
    model.eval()

    token_ids = torch.randint(1, 20, (3, 10))
    lengths = torch.tensor([10, 5, 1])
    logits_before = model(token_ids, lengths)

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    reloaded = _make_model(vocab_size=20, num_classes=5)
    reloaded.load_state_dict(torch.load(path))
    reloaded.eval()

    logits_after = reloaded(token_ids, lengths)
    assert torch.allclose(logits_before, logits_after, atol=1e-6)


def test_overfit_tiny_batch_loss_decreases():
    """Sanity check: the model can memorize a tiny fixed batch in a few steps."""
    torch.manual_seed(0)
    model = _make_model(vocab_size=20, num_classes=5, dropout=0.0)
    model.train()

    token_ids = torch.randint(1, 20, (6, 10))
    lengths = torch.full((6,), 10, dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 3, 4, 0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        logits = model(token_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1
