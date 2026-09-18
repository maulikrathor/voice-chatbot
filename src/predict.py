"""
Inference wrapper around the trained BiLSTM intent classifier.

Will define `IntentPredictor`, which loads the saved artifacts
(model.pt, vocab.json, labels.json, config.json) once and exposes a method
returning the top-k predicted intents and probabilities for a normalized
input text.
"""

# TODO (M3)
