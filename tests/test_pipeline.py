"""
Tests for src.asr.prepare_audio (pure, no Whisper), src.pipeline.VoiceChatbot
text path (load_asr=False), and one slow end-to-end ASR smoke test.
"""

import numpy as np
import pytest

from src.asr import prepare_audio
from src.config import SILENCE_RMS_THRESHOLD, TARGET_SAMPLE_RATE
from src.pipeline import VoiceChatbot
from src.text_preprocessing import normalize_text


def test_prepare_audio_int16_stereo_resampled_to_mono_16khz():
    sample_rate = 44100
    duration_s = 1.0
    num_samples = int(sample_rate * duration_s)

    # A simple sine tone, duplicated across two channels, as int16 PCM --
    # the format Gradio's mic/upload components commonly hand back.
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    stereo = np.stack([tone, tone], axis=1)  # (N, 2)

    audio = prepare_audio(sample_rate, stereo)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    # Resampled from 44.1kHz to 16kHz: length should be close to 16000.
    assert abs(len(audio) - TARGET_SAMPLE_RATE) < 200
    assert np.all(audio >= -1.0) and np.all(audio <= 1.0)


def test_prepare_audio_silence_has_near_zero_rms():
    sample_rate = TARGET_SAMPLE_RATE
    zeros = np.zeros(sample_rate, dtype=np.int16)  # 1 second of silence

    audio = prepare_audio(sample_rate, zeros)
    rms = float(np.sqrt(np.mean(np.square(audio))))

    assert rms < SILENCE_RMS_THRESHOLD


def test_prepare_audio_short_clip_has_expected_duration():
    sample_rate = TARGET_SAMPLE_RATE
    duration_s = 0.2
    num_samples = int(sample_rate * duration_s)
    short_clip = np.zeros(num_samples, dtype=np.int16)

    audio = prepare_audio(sample_rate, short_clip)

    measured_duration = len(audio) / TARGET_SAMPLE_RATE
    assert measured_duration == pytest.approx(duration_s, abs=0.01)


def test_handle_text_end_to_end_without_loading_asr():
    chatbot = VoiceChatbot(load_asr=False)

    result = chatbot.handle_text("what's the weather like today")

    assert result["source"] == "text"
    assert result["transcript"] == "what's the weather like today"
    assert result["intent"] == "weather"
    assert result["asr_status"] is None
    assert isinstance(result["response"], str) and result["response"].strip()


def test_handle_audio_raises_without_asr_loaded():
    chatbot = VoiceChatbot(load_asr=False)
    with pytest.raises(RuntimeError):
        chatbot.handle_audio(TARGET_SAMPLE_RATE, np.zeros(TARGET_SAMPLE_RATE, dtype=np.int16))


def _word_overlap(reference: str, hypothesis: str) -> float:
    """Case/punctuation-insensitive word overlap: |ref words seen in hyp| / |ref words|."""
    ref_words = set(normalize_text(reference).split())
    hyp_words = set(normalize_text(hypothesis).split())
    if not ref_words:
        return 0.0
    return len(ref_words & hyp_words) / len(ref_words)


@pytest.mark.slow
def test_asr_smoke_test_on_librispeech_dummy_sample():
    """
    End-to-end ASR check: transcribe one real sample from the standard HF
    smoke-test dataset and confirm the transcript is non-empty and roughly
    matches the reference (case/punctuation-insensitive word overlap >= 0.6).
    """
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    from src.asr import SpeechRecognizer

    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    # Keep the audio column as raw encoded bytes instead of letting datasets
    # auto-decode it -- the installed `datasets` version requires the extra
    # `torchcodec` dependency for that, which is out of scope here. librosa
    # (already a project runtime dependency) decodes the flac bytes instead.
    dataset = dataset.cast_column("audio", Audio(decode=False))
    sample = dataset[0]
    reference = sample["text"]

    audio_bytes = sample["audio"]["bytes"]
    array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    recognizer = SpeechRecognizer()
    result = recognizer.transcribe(sampling_rate, array)

    print(f"\nreference:  {reference}")
    print(f"hypothesis: {result['text']}")

    assert result["status"] in ("ok", "truncated")
    assert result["text"].strip() != ""

    overlap = _word_overlap(reference, result["text"])
    print(f"word overlap: {overlap:.2f}")
    assert overlap >= 0.6
