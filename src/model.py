"""
BiLSTM intent classifier architecture.

Will define `BiLSTMClassifier`: Embedding(padding_idx=0) -> BiLSTM ->
masked max-pool over time -> Dropout -> Linear(NUM_CLASSES), matching the
architecture frozen in ARCHITECTURE.md section 4.
"""

# TODO (M2)
