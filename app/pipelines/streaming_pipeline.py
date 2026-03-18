import time

from realtime.chunker import VoiceChunker
from realtime.parallel_worker import ParallelChunkProcessor
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

_chunker         = VoiceChunker(silence_trigger_ms=500, min_speech_ms=300, max_chunk_ms=8000)
_transcript_buf  = ""
_translation_buf = ""
_current_target  = "Hindi"


def _process_chunk(wav_path: str) -> dict:
    """
    Runs inside a thread-pool thread.
    ASR + Translation happen here — in parallel with other chunks.
    """
    t0 = time.perf_counter()
    text, src_lang = transcribe(wav_path)
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
        "src_lang":   src_lang,
        "asr_ms":     asr_ms,
        "tr_ms":      tr_ms,
    }


_processor = ParallelChunkProcessor(process_fn=_process_chunk, max_in_flight=8)


def reset_stream():
    global _transcript_buf, _translation_buf
    _chunker.reset()
    _processor.reset()
    _transcript_buf  = ""
    _translation_buf = ""


def run_pipeline(audio, sr: int, target_lang: str):
    global _transcript_buf, _translation_buf, _current_target

    if audio is None:
        return _transcript_buf, _translation_buf, None, ""

    _current_target = target_lang
    t_frame = time.perf_counter()

    chunk = _chunker.push(audio, sr)
    if chunk is not None:
        _processor.push(chunk, sample_rate=16000)

    ready = _processor.drain()
    if not ready:
        return _transcript_buf, _translation_buf, None, ""

    last_translated = ""
    total_asr = 0.0
    total_tr  = 0.0

    for _seq, result in ready:
        if not result.get("text"):
            continue
        _transcript_buf  = (_transcript_buf  + " " + result["text"]).strip()
        _translation_buf = (_translation_buf + " " + result["translated"]).strip()
        last_translated  = result["translated"]
        total_asr += result["asr_ms"]
        total_tr  += result["tr_ms"]

    if not last_translated:
        return _transcript_buf, _translation_buf, None, ""

    t_tts = time.perf_counter()
    speech_file = speak(last_translated)
    tts_ms = round((time.perf_counter() - t_tts) * 1000, 1)

    total_ms = round((time.perf_counter() - t_frame) * 1000, 1)
    latency  = (
        f"ASR {total_asr:.0f} ms  |  Translate {total_tr:.0f} ms  |  "
        f"TTS {tts_ms} ms  |  Total {total_ms} ms  |  "
        f"pool: {_processor.in_flight} in-flight"
    )
    return _transcript_buf, _translation_buf, speech_file, latency
