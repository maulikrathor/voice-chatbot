"""
Speech-to-text via Whisper base.en.

Will define `SpeechRecognizer`: audio preparation (mono, float32, 16 kHz),
a silence/too-short gate, and transcription using the Hugging Face
"openai/whisper-base.en" model.
"""

# TODO (M3)
