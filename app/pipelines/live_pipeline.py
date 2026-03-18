import tempfile
import soundfile as sf
import time

from realtime.chunker import VoiceChunker
from services.asr_travel import transcribe_travel   # medium Whisper – best accuracy without TTS cost
from services.translator import translate

LANGUAGES = {
    "English":   "eng_Latn",
    "Hindi":     "hin_Deva",
    "Bengali":   "ben_Beng",
    "Tamil":     "tam_Taml",
    "Telugu":    "tel_Telu",
    "Kannada":   "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Marathi":   "mar_Deva",
    "Gujarati":  "guj_Gujr",
    "Punjabi":   "pan_Guru",
    "Urdu":      "urd_Arab",
    "Nepali":    "npi_Deva",
    "Odia":      "ory_Orya",
    "Assamese":  "asm_Beng",
    "Sindhi":    "snd_Arab",
    "Sanskrit":  "san_Deva",
}

# ── Per-session state ────────────────────────────────────────────────────────
# For single-user (Gradio default), module-level globals are fine.
# For multi-user: key by session id.

_chunker     = VoiceChunker(silence_trigger_ms=400, min_speech_ms=250, max_chunk_ms=7000)
_transcript  = ""   # accumulated original text
_translation = ""   # accumulated translated text


def reset_live():
    """Call when the user starts a new live session."""
    global _transcript, _translation
    _chunker.reset()
    _transcript  = ""
    _translation = ""


def run_live_pipeline(
    audio,
    sr: int,
    target_lang: str,
) -> tuple[str, str, str]:
    """
    Gradio streaming callback – fires on every mic frame.

    Returns:
        transcript  – growing original text
        translation – growing translated text
        latency_str – timing info string
    """
    global _transcript, _translation

    if audio is None:
        return _transcript, _translation, ""

    t0 = time.perf_counter()

    # ── Feed into VAD chunker ────────────────────────────────────────────────
    chunk = _chunker.push(audio, sr)

    if chunk is None:
        # Still accumulating – return current buffers immediately (no flicker)
        return _transcript, _translation, ""

    # ── Write chunk to temp WAV ──────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, chunk, 16000)
        tmp_path = tmp.name

    # ── ASR (Whisper medium) ─────────────────────────────────────────────────
    t_asr = time.perf_counter()
    text, src_lang = transcribe_travel(tmp_path)
    asr_ms = round((time.perf_counter() - t_asr) * 1000, 1)

    if not text.strip():
        return _transcript, _translation, ""

    # ── Translation ──────────────────────────────────────────────────────────
    t_tr = time.perf_counter()
    tgt_code   = LANGUAGES.get(target_lang, "hin_Deva")
    new_trans  = translate(text.strip(), src_lang, tgt_code)
    tr_ms      = round((time.perf_counter() - t_tr) * 1000, 1)

    # Append both in lockstep — one new line per chunk so columns stay aligned
    _transcript  = (_transcript  + "\n" + text.strip()).lstrip("\n")
    _translation = (_translation + "\n" + new_trans.strip()).lstrip("\n")

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    latency  = f"ASR {asr_ms} ms  |  Translate {tr_ms} ms  |  Total {total_ms} ms"

    return _transcript, _translation, latency
