import time

from realtime.chunker import VoiceChunker
from realtime.parallel_worker import ParallelChunkProcessor
from services.asr_travel import transcribe_travel   # medium Whisper for accuracy
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

_chunker        = VoiceChunker(silence_trigger_ms=450, min_speech_ms=250, max_chunk_ms=8000)
_transcript     = ""
_translation    = ""
_current_target = "Hindi"


def _process_chunk(wav_path: str) -> dict:
    """
    Runs in a thread-pool thread.
    Whisper medium ASR + NLLB translation — fully parallel across chunks.
    """
    t0 = time.perf_counter()
    text, src_lang = transcribe_travel(wav_path)
    asr_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not text.strip():
        return {"text": "", "translated": "", "asr_ms": asr_ms, "tr_ms": 0.0}

    tgt_code   = LANGUAGES.get(_current_target, "hin_Deva")
    t_tr       = time.perf_counter()
    translated = translate(text.strip(), src_lang, tgt_code)
    tr_ms      = round((time.perf_counter() - t_tr) * 1000, 1)

    return {
        "text":       text.strip(),
        "translated": translated.strip(),
        "asr_ms":     asr_ms,
        "tr_ms":      tr_ms,
    }


_processor = ParallelChunkProcessor(process_fn=_process_chunk, max_in_flight=8)


def reset_live():
    global _transcript, _translation
    _chunker.reset()
    _processor.reset()
    _transcript  = ""
    _translation = ""


def run_live_pipeline(audio, sr: int, target_lang: str):
    """
    Gradio streaming callback — no TTS.

    Returns (transcript, translation, latency_str).
    Both text boxes grow one line per chunk, always in sync.
    """
    global _transcript, _translation, _current_target

    if audio is None:
        return _transcript, _translation, ""

    _current_target = target_lang
    t_frame = time.perf_counter()

    chunk = _chunker.push(audio, sr)
    if chunk is not None:
        _processor.push(chunk, sample_rate=16000)

    ready = _processor.drain()
    if not ready:
        return _transcript, _translation, ""

    total_asr = 0.0
    total_tr  = 0.0
    new_lines = 0

    for _seq, result in ready:
        if not result.get("text"):
            continue
        # Append line-by-line so both columns stay aligned
        _transcript  = (_transcript  + "\n" + result["text"]).lstrip("\n")
        _translation = (_translation + "\n" + result["translated"]).lstrip("\n")
        total_asr += result["asr_ms"]
        total_tr  += result["tr_ms"]
        new_lines  += 1

    if new_lines == 0:
        return _transcript, _translation, ""

    total_ms = round((time.perf_counter() - t_frame) * 1000, 1)
    latency  = (
        f"ASR {total_asr:.0f} ms  |  Translate {total_tr:.0f} ms  |  "
        f"Total {total_ms} ms  |  pool: {_processor.in_flight} in-flight"
    )
    return _transcript, _translation, latency
