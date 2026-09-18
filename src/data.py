"""
Loading of the CLINC150 "plus" dataset.

Tries the Hugging Face Hub first (dataset id "clinc/clinc_oos", then the
older id "clinc_oos", both with config "plus"). If neither is reachable,
falls back to the raw "data_oos_plus.json" file from the clinc/oos-eval
GitHub repository and reconstructs the same (text, label_id) structure.

The out-of-scope class is always named "oos" in id2label, regardless of
which loading path was used.
"""

import json
import urllib.request

from datasets import load_dataset

# Hugging Face dataset ids to try, in order, both under the "plus" config.
_HF_DATASET_IDS = ("clinc/clinc_oos", "clinc_oos")

# Fallback: raw CLINC150 "plus" data file (in-scope + out-of-scope combined)
# from the original paper's GitHub repository.
_GITHUB_JSON_URL = (
    "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json"
)

Split = list[tuple[str, int]]


def _pairs_from_hf_split(split) -> Split:
    """Turn a Hugging Face Dataset split into a list of (text, label_id)."""
    texts = split["text"]
    labels = split["intent"]
    return list(zip(texts, labels))


def _load_from_huggingface():
    """
    Try each known Hugging Face dataset id for CLINC150 "plus".

    Returns (train, val, test, id2label, label2id) on success, or None if
    none of the dataset ids could be loaded (e.g. no network access).
    """
    for dataset_id in _HF_DATASET_IDS:
        try:
            dataset = load_dataset(dataset_id, "plus")
        except Exception:
            continue

        label_names = dataset["train"].features["intent"].names
        id2label = dict(enumerate(label_names))
        label2id = {name: idx for idx, name in id2label.items()}

        train = _pairs_from_hf_split(dataset["train"])
        val = _pairs_from_hf_split(dataset["validation"])
        test = _pairs_from_hf_split(dataset["test"])

        print(f"[data] Loaded CLINC150 plus from Hugging Face Hub dataset '{dataset_id}'.")
        return train, val, test, id2label, label2id

    return None


def _load_from_github_json():
    """
    Fall back to the raw data_oos_plus.json file from clinc/oos-eval.

    That file stores in-scope examples under "train"/"val"/"test" (as
    [text, label] pairs) and the additional out-of-scope examples under
    "oos_train"/"oos_val"/"oos_test" (as [text, "oos"] pairs). This function
    merges each in-scope split with its matching oos split, matching the
    combined "plus" split sizes served by the Hugging Face dataset.

    Returns (train, val, test, id2label, label2id) on success, or None if
    the file could not be downloaded or parsed.
    """
    try:
        with urllib.request.urlopen(_GITHUB_JSON_URL, timeout=30) as response:
            raw = json.load(response)
    except Exception:
        return None

    in_scope_labels = sorted({label for _, label in raw["train"]})
    label2id = {label: idx for idx, label in enumerate(in_scope_labels)}
    label2id["oos"] = len(label2id)
    id2label = {idx: label for label, idx in label2id.items()}

    def _combine(in_scope_key: str, oos_key: str) -> Split:
        pairs = [(text, label2id[label]) for text, label in raw[in_scope_key]]
        pairs += [(text, label2id["oos"]) for text, _ in raw[oos_key]]
        return pairs

    train = _combine("train", "oos_train")
    val = _combine("val", "oos_val")
    test = _combine("test", "oos_test")

    print(f"[data] Loaded CLINC150 plus from GitHub fallback '{_GITHUB_JSON_URL}'.")
    return train, val, test, id2label, label2id


def load_clinc():
    """
    Load the CLINC150 "plus" dataset.

    Returns:
        train: list of (text, label_id) for the training split.
        val: list of (text, label_id) for the validation split.
        test: list of (text, label_id) for the test split.
        id2label: dict mapping label_id -> intent name (includes "oos").
        label2id: dict mapping intent name -> label_id (includes "oos").

    Raises:
        RuntimeError: if the dataset could not be loaded from either the
            Hugging Face Hub or the GitHub fallback.
    """
    result = _load_from_huggingface()
    if result is None:
        result = _load_from_github_json()
    if result is None:
        raise RuntimeError(
            "Could not load the CLINC150 'plus' dataset from the Hugging Face "
            "Hub (tried: " + ", ".join(_HF_DATASET_IDS) + ") or from the "
            "GitHub fallback (" + _GITHUB_JSON_URL + "). Check network "
            "connectivity and try again."
        )

    train, val, test, id2label, label2id = result
    return train, val, test, id2label, label2id
