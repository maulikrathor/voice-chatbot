# PROJECT_SPEC — Voice-Enabled Intent Chatbot

## 1. Assignment requirements

| ID | Requirement |
|----|-------------|
| R1 | Accept voice input from the user |
| R2 | Convert speech to text using a suitable speech recognition technique |
| R3 | Process text with a deep-learning chatbot / intent classification model |
| R4 | Generate an appropriate response |
| R5 | Display both the recognized speech and the chatbot response |
| R6 | Deploy online, publicly accessible; submit live link |
| R7 | Submit source code + brief report (dataset, architecture, methodology, results) |

Deadline: 30 September 2026. Weight: 20 marks. Time budget: 1–2 days.

## 2. Objective

Build the simplest technically defensible voice chatbot that satisfies R1–R7, works reliably online, and is easy to explain in a viva.

## 3. Selected architecture

Gradio web app hosted on Hugging Face Spaces (free CPU tier).
Voice → **Whisper `base.en`** (pretrained transformer ASR, runs on server, no API key) → text normalization → **BiLSTM intent classifier trained on CLINC150 "plus"** (151 classes incl. out-of-scope) → **template response map** with a confidence-threshold fallback → UI shows transcript, intent, confidence, and response.

Training runs locally on CPU; only small artifacts (~5 MB) are deployed.

## 4. Functional requirements

- FR1: Record audio from the browser microphone (R1).
- FR2: Accept an uploaded audio file (wav/mp3/m4a) as an alternative (R1).
- FR3: Accept typed text as a fallback when no mic is available.
- FR4: Transcribe audio to English text with Whisper (R2).
- FR5: Reject silent or too-short audio with a clear message instead of transcribing.
- FR6: Normalize text identically at training and inference time.
- FR7: Predict intent + confidence with the trained BiLSTM (R3).
- FR8: If the predicted intent is `oos` or confidence < tuned threshold, return a fallback response.
- FR9: Map the intent to a response; dynamic answers for time/date (Asia/Kolkata) (R4).
- FR10: Display recognized text, predicted intent, confidence, and bot response; keep a chat history (R5).
- FR11: Provide clickable example queries.

## 5. Non-functional requirements

- NFR1: No external API keys or paid services at runtime.
- NFR2: End-to-end response ≤ ~5 s for a 5 s utterance on Spaces CPU (models loaded once at startup).
- NFR3: Deployed artifacts < 10 MB each (no Git LFS needed).
- NFR4: Readable, modular code; preprocessing shared between train and inference.
- NFR5: Reproducible training (fixed seeds, saved config and metrics).
- NFR6: Runs in any modern browser over HTTPS.

## 6. Constraints / out of scope

Out of scope: RAG, LLM APIs, agents, vector DBs, databases, Docker, custom JS frontends, user accounts, large-scale training, MLOps tooling.

## 7. Acceptance criteria

- [ ] BiLSTM test in-scope accuracy ≥ 85% (report actual numbers, incl. OOS recall and macro-F1).
- [ ] 8 of 10 self-recorded in-domain utterances produce the correct intent end-to-end.
- [ ] Silent clip produces a "didn't catch that" message, not a hallucinated transcript.
- [ ] Out-of-scope query (e.g. "who won the 1998 world cup") triggers the fallback.
- [ ] Public HF Space link works from a second device/browser without login.
- [ ] Repo contains source, artifacts, README, and REPORT.md.
- [ ] All `pytest` tests pass locally.
