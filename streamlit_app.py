"""
Streamlit frontend for the voice chatbot (Streamlit Community Cloud's main
file). Reuses the same VoiceChatbot pipeline (src/pipeline.py) and the same
formatting helpers (src/formatting.py) as the Gradio app (app.py, src/ui.py)
so both frontends transcribe/classify/respond identically and show
identical wording -- only the widgets differ.

This module intentionally does not import src.ui or gradio: the Streamlit
deployment's requirements.txt has no gradio in it (see requirements.txt),
so nothing here may depend on it.
"""

import traceback

import streamlit as st

from src.asr import bytes_to_audio
from src.config import ARTIFACTS_DIR, WHISPER_MODEL_ID
from src.formatting import TEXT_EXAMPLES, format_understanding, load_about_markdown, top3_dict
from src.pipeline import VoiceChatbot

_EMPTY_TEXT_MESSAGE = "Please type a message first."
_EMPTY_AUDIO_MESSAGE = "Please record or upload some audio first."
_ERROR_MESSAGE = "Something went wrong on my end. Please try again."

st.set_page_config(page_title="Voice Intent Chatbot", page_icon="🎤")


@st.cache_resource(show_spinner="Loading models (Whisper + BiLSTM) -- this can take a minute on first run...")
def load_bot() -> VoiceChatbot:
    print(f"Streamlit app starting; WHISPER_MODEL_ID={WHISPER_MODEL_ID}")
    return VoiceChatbot(load_asr=True)


bot = load_bot()


# --- session state -----------------------------------------------------------

def _init_state():
    defaults = {
        "history": [],  # list of {"role": "user"/"assistant", "content": str}
        "you_said": "",
        "understanding": "",
        "top3": {},
        "response": "",
        "last_audio_id": None,  # st.audio_input UploadedFile.file_id already processed
        "reset_counter": 0,  # bumped by Clear to force-remount widgets with fresh state
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


# --- processing helpers -------------------------------------------------------
# Mirrors src.ui.process_text / process_voice, but stores results directly in
# st.session_state instead of returning a tuple -- that's the natural shape
# for Streamlit's rerun-on-interaction model.

def _run_text(text: str) -> None:
    if not text or not text.strip():
        st.session_state.understanding = _EMPTY_TEXT_MESSAGE
        return

    try:
        result = bot.handle_text(text)
    except Exception:
        traceback.print_exc()
        st.session_state.you_said = text
        st.session_state.understanding = _ERROR_MESSAGE
        st.session_state.top3 = {}
        st.session_state.response = ""
        return

    st.session_state.you_said = result["transcript"]
    st.session_state.understanding = format_understanding(result, bot.predictor.threshold)
    st.session_state.top3 = top3_dict(result["top3"])
    st.session_state.response = result["response"]
    st.session_state.history.append({"role": "user", "content": text})
    st.session_state.history.append({"role": "assistant", "content": result["response"]})


def _run_audio(raw_bytes: bytes) -> None:
    try:
        sample_rate, data = bytes_to_audio(raw_bytes)
        result = bot.handle_audio(sample_rate, data)
    except Exception:
        traceback.print_exc()
        st.session_state.understanding = _ERROR_MESSAGE
        st.session_state.top3 = {}
        st.session_state.response = ""
        return

    transcript = result["transcript"]
    st.session_state.you_said = transcript
    st.session_state.understanding = format_understanding(result, bot.predictor.threshold)
    st.session_state.top3 = top3_dict(result["top3"])
    st.session_state.response = result["response"]
    user_turn = f"🎤 {transcript}" if transcript else "🎤 (no speech detected)"
    st.session_state.history.append({"role": "user", "content": user_turn})
    st.session_state.history.append({"role": "assistant", "content": result["response"]})


def _clear() -> None:
    st.session_state.history = []
    st.session_state.you_said = ""
    st.session_state.understanding = ""
    st.session_state.top3 = {}
    st.session_state.response = ""
    st.session_state.last_audio_id = None
    st.session_state.reset_counter += 1  # gives widgets below fresh keys -> fresh state


# --- UI ------------------------------------------------------------------------

st.title("Voice Intent Chatbot")
st.markdown(
    "Speak into your microphone (or upload/type) and the bot will transcribe, "
    "classify, and respond. It recognizes 150 everyday intents across 10 domains "
    "(banking, travel, small talk, utility, and more)."
)

_key_suffix = st.session_state.reset_counter  # bump on Clear to reset these widgets

st.subheader("Speak")
mic_audio = st.audio_input("Record a message", key=f"mic_{_key_suffix}")
if mic_audio is not None and mic_audio.file_id != st.session_state.last_audio_id:
    st.session_state.last_audio_id = mic_audio.file_id
    _run_audio(mic_audio.getvalue())

uploaded_file = st.file_uploader(
    "...or upload an audio file", type=["wav", "flac", "ogg", "mp3"], key=f"upload_{_key_suffix}"
)
if st.button("Send audio"):
    if uploaded_file is None:
        st.session_state.understanding = _EMPTY_AUDIO_MESSAGE
    else:
        _run_audio(uploaded_file.getvalue())

st.subheader("Or type")
text_query = st.text_input(
    "Type a message (press Enter to send)",
    key=f"text_query_{_key_suffix}",
    placeholder="e.g. book me a flight to chicago",
    on_change=lambda: _run_text(st.session_state[f"text_query_{_key_suffix}"]),
)
if st.button("Send text"):
    _run_text(text_query)

st.caption("Try one of these:")
example_cols = st.columns(3)
for i, example in enumerate(TEXT_EXAMPLES):
    if example_cols[i % 3].button(example, key=f"example_{i}_{_key_suffix}"):
        _run_text(example)

st.divider()

st.text_input("You said", value=st.session_state.you_said, disabled=True)
st.markdown(f"**Understanding:** {st.session_state.understanding}" if st.session_state.understanding else "**Understanding:**")

if st.session_state.top3:
    st.caption("Top-3 intents")
    for label, prob in st.session_state.top3.items():
        st.progress(prob, text=f"{label}: {prob * 100:.1f}%")

st.text_input("Bot response", value=st.session_state.response, disabled=True)

st.subheader("Conversation")
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

st.button("Clear", on_click=_clear)

with st.expander("About this model"):
    st.markdown(load_about_markdown(ARTIFACTS_DIR))
