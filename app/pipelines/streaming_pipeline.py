import tempfile
import soundfile as sf
import time

from realtime.chunker import VoiceChunker
from services.asr import transcribe
from services.translator import translate
from services.tts_engine import speak

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

# One chunker instance per Gradio session is not trivially achievable with
# the current stateless callback pattern, so we keep a single global chunker.
# For multi-user deployments, key this dict by session_id.
_chunker = VoiceChunker(silence_trigger_ms=500, min_speech_ms=300, max_chunk_ms=8000)
_transcript_buffer = ""


def reset_stream():
    """Call when a new recording session starts."""
    global _transcript_buffer
    _chunker.reset()
    _transcript_buffer = ""


def run_pipeline(audio, sr, target_lang):
    """
    Called on every streaming audio frame from Gradio.

    Returns (transcript, translation, speech_file, latency_str).
    Returns previous state when no complete chunk is ready yet.
    """
    global _transcript_buffer

    if audio is None:
        return _transcript_buffer, "", None, ""

    t0 = time.perf_counter()

    # Feed into chunker; get a complete speech chunk or None
    chunk = _chunker.push(audio, sr)

    if chunk is None:
        # Not enough speech yet – return buffered state with no audio
        return _transcript_buffer, "", None, ""

    # --- Write chunk to temp WAV ---
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, chunk, 16000)
        tmp_path = tmp.name

    # --- ASR ---
    t_asr = time.perf_counter()
    text, src_lang = transcribe(tmp_path)
    asr_ms = round((time.perf_counter() - t_asr) * 1000, 1)

    if not text.strip():
        return _transcript_buffer, "", None, ""

    _transcript_buffer = (_transcript_buffer + " " + text).strip()

    # --- Translation ---
    t_tr = time.perf_counter()
    tgt_code = LANGUAGES.get(target_lang, "hin_Deva")
    translated = translate(text, src_lang, tgt_code)   # translate only the new chunk
    tr_ms = round((time.perf_counter() - t_tr) * 1000, 1)

    # --- TTS ---
    t_tts = time.perf_counter()
    speech_file = speak(translated)
    tts_ms = round((time.perf_counter() - t_tts) * 1000, 1)

    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    latency = f"ASR {asr_ms} ms  |  Translate {tr_ms} ms  |  TTS {tts_ms} ms  |  Total {total_ms} ms"
    return _transcript_buffer, translated, speech_file, latency
