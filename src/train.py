"""
Training script for the BiLSTM intent classifier.

Will load CLINC150 plus via src.data, build the Vocab from the training
split, train BiLSTMClassifier with early stopping on validation accuracy,
tune the OOS confidence threshold on the validation split, and save
model.pt, vocab.json, labels.json, config.json, and metrics.json to
artifacts/.
"""

# TODO (M2)
