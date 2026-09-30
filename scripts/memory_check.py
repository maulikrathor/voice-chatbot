"""
Report peak resident memory for one text query + one audio query through
the full VoiceChatbot pipeline (BiLSTM + Whisper loaded). Used to check
whether whisper-base.en fits Streamlit Community Cloud's ~1 GB RAM cap, or
whether we need to drop to whisper-tiny.en.

Usage:
    python scripts/memory_check.py
    WHISPER_MODEL_ID=openai/whisper-tiny.en python scripts/memory_check.py
"""

import sys
from pathlib import Path

# Allow running as `python scripts/memory_check.py` from the repo root
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa

from src.config import REPO_ROOT, WHISPER_MODEL_ID
from src.pipeline import VoiceChatbot

SAMPLE_AUDIO = REPO_ROOT / "samples" / "04_book_flight.wav"


def _peak_memory_mb() -> tuple[float, str] | tuple[None, str]:
    """
    Peak resident memory for this process in MB, and a label describing
    what was measured (the exact metric differs by platform). Returns
    (None, reason) if no reliable method is available.
    """
    try:
        import resource  # Unix only -- not available on Windows.

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports ru_maxrss in KB; macOS reports it in bytes.
        peak_mb = peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)
        return peak_mb, "peak RSS (resource.ru_maxrss)"
    except ImportError:
        pass

    if sys.platform == "win32":
        # No `resource` module on Windows and no psutil dependency here
        # (kept out of requirements.txt), so fall back to the Win32 API
        # directly via ctypes. PeakWorkingSetSize is Windows' closest
        # analogue to Linux's peak RSS: physical memory ever charged to
        # this process's working set.
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # restype/argtypes must be set explicitly: ctypes defaults to
        # `int`, which truncates the 64-bit pseudo-handle GetCurrentProcess
        # returns and makes GetProcessMemoryInfo fail silently.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return counters.PeakWorkingSetSize / (1024 * 1024), "peak working set size (Windows RSS analogue)"
        return None, "GetProcessMemoryInfo call failed"

    return None, f"no reliable peak-memory method for platform {sys.platform!r}"


def main() -> None:
    print(f"WHISPER_MODEL_ID = {WHISPER_MODEL_ID}")

    bot = VoiceChatbot(load_asr=True)

    text_result = bot.handle_text("what's the weather like today")
    print(f"Text query  -> intent={text_result['intent']!r} confidence={text_result['confidence']:.3f}")

    data, sample_rate = librosa.load(SAMPLE_AUDIO, sr=None, mono=False)
    if data.ndim == 2:
        data = data.T
    audio_result = bot.handle_audio(sample_rate, data)
    print(f"Audio query -> intent={audio_result['intent']!r} confidence={audio_result['confidence']:.3f}")

    peak_mb, label = _peak_memory_mb()
    if peak_mb is None:
        print(f"Could not measure peak memory: {label}")
    else:
        print(f"{label}: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
