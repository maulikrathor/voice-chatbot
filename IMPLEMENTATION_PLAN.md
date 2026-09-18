# IMPLEMENTATION_PLAN — Voice-Enabled Intent Chatbot

## 1. Repository structure

```
voice-chatbot/
├── app.py                     # Gradio UI (M4)
├── requirements.txt           # runtime deps only (Space)
├── requirements-train.txt     # training/testing deps (local)
├── README.md                  # M5 (Space YAML header) / M6 (full)
├── REPORT.md                  # M6
├── PROJECT_SPEC.md
├── ARCHITECTURE.md
├── IMPLEMENTATION_PLAN.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py              # M1
│   ├── data.py                # M1
│   ├── text_preprocessing.py  # M1
│   ├── model.py               # M2
│   ├── train.py               # M2
│   ├── evaluate.py            # M2
│   ├── predict.py             # M3
│   ├── asr.py                 # M3
│   ├── responses.py           # M3
│   └── pipeline.py            # M3
├── data/
│   └── responses.json         # M3: 151 entries, 1–3 variants each
├── artifacts/                 # M2 outputs (committed)
├── samples/                   # user-recorded test clips (manual)
├── scripts/
│   └── dataset_stats.py       # M1
└── tests/
    ├── test_data.py           # M1
    ├── test_preprocessing.py  # M1
    ├── test_model.py          # M2
    ├── test_predict.py        # M3
    ├── test_responses.py      # M3
    └── test_pipeline.py       # M3
```

Stub files for later milestones contain only a module docstring and a `# TODO (Mx)` line until their milestone.

## 2. Milestones

### M1 — Scaffold + data pipeline (Day 1, ~1.5 h)
- Repo structure, `.gitignore`, both requirements files, venv, `git init`.
- `config.py`: paths, seed, `MAX_LEN=32`, `MIN_FREQ=2`, special tokens.
- `data.py`: load CLINC150 plus (try `clinc/clinc_oos` then `clinc_oos` on HF Hub, config `plus`; fallback: raw JSON `data_oos_plus.json` from the clinc/oos-eval GitHub repo). Return train/val/test as lists of `(text, label_id)` plus `id2label`/`label2id`.
- `text_preprocessing.py`: `normalize_text`, `tokenize`, `Vocab` (build from train, encode with pad/truncate, save/load JSON).
- `scripts/dataset_stats.py`: prints split sizes, class count, vocab size, OOV rate on val.
- **Checkpoint:** tests pass; split sizes 15,250 / 3,100 / 5,500; 151 labels incl. `oos`.

### M2 — Model, training, evaluation (Day 1, ~2 h)
- `model.py` (BiLSTM, masked max-pool), `train.py`, `evaluate.py`.
- Save `model.pt`, `vocab.json`, `labels.json`, `config.json` (incl. tuned threshold), `metrics.json`.
- **Checkpoint:** shape tests pass; test in-scope accuracy ≥ 85% (if < 80%, stop and investigate before M3); artifacts < 10 MB.

### M3 — Inference modules (Day 1, ~2 h)
- `predict.py`, `asr.py` (audio prep + silence gate + Whisper `openai/whisper-base.en`), `responses.py` + `data/responses.json`, `pipeline.py`.
- Dynamic handlers: time, date (Asia/Kolkata). Fallback for `oos`/low confidence.
- **Checkpoint:** every label has a response; ASR smoke test on a sample from `hf-internal-testing/librispeech_asr_dummy`; silence → "didn't catch that"; text pipeline returns correct intents for ~10 hand-written queries.

### M4 — Gradio app (Day 2, ~1.5 h)
- Mic + upload + text inputs; outputs: transcript, intent + confidence, response, chat history; example queries; models loaded once.
- **Checkpoint:** `python app.py` works locally; user tests 10 own recordings (≥ 8 correct).

### M5 — Deployment (Day 2, ~1 h)
- Space README YAML header; CPU-only torch in `requirements.txt`; push to Space.
- **Checkpoint:** build succeeds; live link works on a second device.

### M6 — Docs + submission (Day 2, ~1.5 h)
- Full README (setup, run, link, results).
- REPORT.md (2–4 pages): intro, dataset, architecture, methodology, results table, sample interactions, limitations, conclusion.
- **Checkpoint:** acceptance criteria in PROJECT_SPEC all ticked.

## 3. Manual tasks (user)
- Create HF account + Space (Gradio SDK, free CPU, public); create a write token for git push.
- Record ~10 test clips into `samples/`.
- Test the live link from a phone; take screenshots for the report.
- Open the Space shortly before evaluation (free Spaces sleep when idle).

## 4. Deployment steps (M5)
1. Add YAML header to `README.md`: `sdk: gradio`, `sdk_version` = locally tested Gradio version, `app_file: app.py`, `python_version: "3.11"`.
2. `requirements.txt`: `--extra-index-url https://download.pytorch.org/whl/cpu` then `torch`, `transformers`, `gradio`, `librosa`, `numpy` (pinned to tested versions).
3. `git remote add space https://huggingface.co/spaces/<user>/<space>`; push.
4. Watch build logs; fix; verify mic, upload, text, fallback on the live URL.
5. Optionally mirror the code to GitHub for the source-code submission.

## 5. Definition of done
- Live public link works end-to-end from a second device.
- All tests pass locally.
- Metrics recorded in `artifacts/metrics.json` and in the report.
- README + REPORT.md complete; source pushed.
