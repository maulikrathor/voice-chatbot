# Project Instructions

## Project
Voice-Enabled Chatbot for Speech and Language Processing Lab Assessment.

## Objective
Build and deploy a public online voice chatbot that:
1. Accepts microphone audio.
2. Converts speech to text using Whisper.
3. Classifies the transcript using a trained BiLSTM intent classifier.
4. Generates an appropriate response.
5. Displays the recognized speech and chatbot response.
6. Is publicly deployable on Hugging Face Spaces.

## Priority

Working and reliable > sophisticated.

This is a 20-mark academic lab assignment with a 1–2 day implementation window.

Do NOT overengineer the project.

Do NOT introduce:
- RAG
- LangGraph
- agents
- databases
- external LLM APIs
- unnecessary backend/frontend separation
- Docker unless deployment absolutely requires it
- unnecessary dependencies

## Architecture

Microphone
→ Whisper base.en
→ transcript normalization
→ BiLSTM intent classifier
→ intent/confidence
→ response mapping
→ Gradio UI

## Engineering Rules

- Follow PROJECT_SPEC.md.
- Follow ARCHITECTURE.md.
- Follow IMPLEMENTATION_PLAN.md.
- Implement one milestone at a time.
- Do not jump ahead to later milestones.
- Run tests after each milestone.
- Do not modify the frozen architecture without identifying a genuine blocker.
- Prefer simple, deterministic implementations.
- Reuse the same preprocessing logic during training and inference.
- Keep deployment in mind from the beginning.
- Do not fabricate test results or model metrics.

## Workflow

At the end of every milestone, report:
- files created/modified
- commands executed
- tests executed
- test results
- model/training results where applicable
- current project state
- known issues
- blockers
- recommended next milestone

Do not begin the next milestone until explicitly instructed.