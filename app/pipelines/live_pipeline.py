import tempfile
import time
import soundfile as sf

from realtime.rolling_buffer import RollingBuffer
from realtime.parallel_worker import ParallelChunkProcessor
from services.asr import transcribe                     
from services.asr_travel import transcribe_travel       
from services.translator import translate
from services.transcript_formatter import TranscriptFormatter

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

_buffer         = RollingBuffer()
_fmt            = TranscriptFormatter()
_current_target = "Hindi"
_partial_text   = ""   # the live in-progress line (not yet committed)


# ── Partial worker (Whisper small, no translation) ────────────────────────────

def _run_partial(wav_path: str) -> dict:
    text, _ = transcribe(wav_path)
    return {"text": text.strip(), "type": "partial"}


# ── Commit worker (Whisper medium + translation) ──────────────────────────────

def _run_commit(wav_path: str) -> dict:
    t0 = time.perf_counter()
    text, src_lang = transcribe_travel(wav_path)
    asr_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not text.strip():
        return {"text": "", "translated": "", "asr_ms": asr_ms, "tr_ms": 0.0, "type": "commit"}

    tgt_code   = LANGUAGES.get(_current_target, "hin_Deva")
    t_tr       = time.perf_counter()
    translated = translate(text.strip(), src_lang, tgt_code)
    tr_ms      = round((time.perf_counter() - t_tr) * 1000, 1)

    return {
        "text":       text.strip(),
        "translated": translated.strip(),
        "asr_ms":     asr_ms,
        "tr_ms":      tr_ms,
        "type":       "commit",
    }


# Two separate pools — partial needs low latency, commit needs accuracy
_partial_pool = ParallelChunkProcessor(process_fn=_run_partial, max_in_flight=2)
_commit_pool  = ParallelChunkProcessor(process_fn=_run_commit,  max_in_flight=8)


def reset_live():
    global _partial_text, _current_target
    _buffer.reset()
    _fmt.reset()
    _partial_pool.reset()
    _commit_pool.reset()
    _partial_text   = ""
    _current_target = "Hindi"


def _write_wav(chunk) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, chunk, 16000)
        return tmp.name


def run_live_pipeline(audio, sr: int, target_lang: str):
    """
    Gradio streaming callback — no TTS.

    Returns (transcript_display, translation_display, latency_str).

    Transcript box:
      - committed lines: clean, normal weight
      - last line: partial / in-progress, marked with trailing underscore

    Translation box:
      - mirrors committed lines
      - shows "…" on the partial line (translation only on commit)
    """
    global _partial_text, _current_target

    if audio is None:
        return _build_display()

    _current_target = target_lang
    t_frame = time.perf_counter()

    # 1. Feed into dual-stream buffer
    partial_chunk, commit_chunk = _buffer.push(audio, sr)

    # 2. Submit to respective pools
    if partial_chunk is not None:
        _partial_pool.push(partial_chunk, sample_rate=16000)
    if commit_chunk is not None:
        _commit_pool.push(commit_chunk, sample_rate=16000)

    # 3. Drain partial results (update the live line, no translation)
    for _seq, result in _partial_pool.drain():
        if result.get("text"):
            _partial_text = result["text"]

    # 4. Drain commit results (add permanent line + translation)
    total_asr = 0.0
    total_tr  = 0.0
    committed = 0

    for _seq, result in _commit_pool.drain():
        if not result.get("text"):
            continue
        _fmt.push(result["text"], result.get("translated", ""))
        _partial_text = ""   # clear partial — the committed line replaces it
        total_asr += result.get("asr_ms", 0)
        total_tr  += result.get("tr_ms", 0)
        committed += 1

    transcript_display, translation_display = _build_display()

    total_ms = round((time.perf_counter() - t_frame) * 1000, 1)
    if committed or partial_chunk is not None:
        latency = (
            f"ASR {total_asr:.0f} ms  |  Translate {total_tr:.0f} ms  |  "
            f"Frame {total_ms} ms  |  "
            f"partial {_partial_pool.in_flight}  commit {_commit_pool.in_flight} in-flight"
        )
    else:
        latency = ""

    return transcript_display, translation_display, latency


def _build_display():
    """
    Assemble what goes in each textbox.

    Transcript: committed lines (normal) + partial line (with trailing _)
    Translation: mirrored committed translations + ellipsis while partial
    """
    t_committed, tr_committed = _fmt.display()

    if _partial_text:
        # Append the live in-progress line visually
        transcript_display  = (t_committed  + "\n" + _partial_text + "_").lstrip("\n")
        translation_display = (tr_committed + "\n" + "…").lstrip("\n")
    else:
        transcript_display  = t_committed
        translation_display = tr_committed

    return transcript_display, translation_display
