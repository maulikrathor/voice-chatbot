"""
BiLSTM intent classifier architecture.

Fixed by ARCHITECTURE.md section 4:

    token ids (B, 32)
     -> Embedding(V, 128, padding_idx=0)
     -> BiLSTM(hidden=128, 1 layer, bidirectional) -> (B, 32, 256)
     -> masked max-pool over time -> (B, 256)
     -> Dropout(0.3) -> Linear(256, 151) -> logits
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMClassifier(nn.Module):
    """Embedding -> BiLSTM -> masked max-pool -> Dropout -> Linear."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) int64 padded token ids.
            lengths: (B,) int64 true (pre-padding) sequence lengths.

        Returns:
            logits: (B, num_classes) float32.
        """
        embedded = self.embedding(token_ids)  # (B, L, E)

        # Clamp to a minimum of 1: pack_padded_sequence cannot handle a
        # zero-length sequence, and an input that tokenized to nothing (e.g.
        # punctuation-only text) must still produce a valid, finite forward
        # pass rather than crash.
        safe_lengths = lengths.clamp(min=1)

        # Packing tells the LSTM the true length of every sequence, so it
        # stops advancing a sequence once it reaches that length instead of
        # continuing to consume <pad> embeddings. This matters most for the
        # backward direction of the BiLSTM, which otherwise would *start*
        # by reading trailing padding before ever reaching real tokens.
        # enforce_sorted=False lets lengths be in any order (we don't sort
        # the batch by length ourselves).
        packed = pack_padded_sequence(
            embedded, safe_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)

        # Undo packing back into a dense (B, L, 2*H) tensor. Positions beyond
        # each sequence's true length (i.e. padded positions) are filled
        # with zeros by pad_packed_sequence.
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=token_ids.size(1)
        )

        # Masked max-pool over time: without masking, the zeros that
        # pad_packed_sequence writes into padded positions could still win
        # the max (whenever every real activation at that feature is
        # negative), letting padding leak into the pooled representation.
        # Setting padded positions to -inf before pooling rules that out.
        is_real_token = (token_ids != self.pad_idx).unsqueeze(-1)  # (B, L, 1)
        masked_output = output.masked_fill(~is_real_token, float("-inf"))
        pooled, _ = masked_output.max(dim=1)  # (B, 2*H)

        # A row with no real tokens at all (fully padded, only reachable via
        # the length-clamping above) has -inf at every feature after
        # masking; replace with zeros so dropout/linear never sees inf/nan.
        pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)

        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits
