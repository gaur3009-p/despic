import numpy as np
import librosa
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
import time

SAMPLE_RATE = 16000

_vad = load_silero_vad()


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def _vad_timestamps(audio: np.ndarray):
    _vad.reset_states()
    return get_speech_timestamps(
        torch.from_numpy(audio), _vad, sampling_rate=SAMPLE_RATE
    )


class RollingBuffer:
    """
    Stateful dual-stream audio buffer.

    Every push() call:
      - Accumulates audio
      - Returns (partial_chunk, commit_chunk) where either can be None

    partial_chunk — the current speech region re-sliced every PARTIAL_INTERVAL_MS.
                    Always the audio from last speech onset to now.
                    Fires frequently so Whisper small can show words as spoken.

    commit_chunk  — a completed utterance (silence detected after speech).
                    Fires once per sentence/phrase. Used for clean ASR + translate.
    """

    PARTIAL_INTERVAL_MS  = 800   # re-transcribe partial every 800 ms of new audio
    SILENCE_COMMIT_MS    = 500   # commit after 500 ms silence
    MIN_SPEECH_MS        = 250   # ignore < 250 ms blips
    MAX_CHUNK_MS         = 8000  # hard cap

    def __init__(self):
        self._buf: np.ndarray = np.array([], dtype=np.float32)
        self._speech_start: int | None = None
        self._last_partial_at: int = 0   # sample index of last partial fire

    def reset(self):
        self._buf = np.array([], dtype=np.float32)
        self._speech_start = None
        self._last_partial_at = 0

    def push(self, audio: np.ndarray, sr: int):
        """
        Returns (partial_chunk_or_None, commit_chunk_or_None).
        """
        audio = _resample(audio, sr)
        self._buf = np.concatenate([self._buf, audio])

        timestamps = _vad_timestamps(self._buf)

        # ── No speech at all ──────────────────────────────────────────────────
        if not timestamps:
            tail = int(self.SILENCE_COMMIT_MS * SAMPLE_RATE / 1000)
            self._buf = self._buf[-tail:]
            self._speech_start   = None
            self._last_partial_at = 0
            return None, None

        first_start = timestamps[0]["start"]
        last_end    = timestamps[-1]["end"]

        if self._speech_start is None:
            self._speech_start = first_start

        samples_since_speech = len(self._buf) - last_end
        speech_len           = last_end - self._speech_start
        samples_since_partial = len(self._buf) - self._last_partial_at

        silence_threshold = int(self.SILENCE_COMMIT_MS  * SAMPLE_RATE / 1000)
        min_speech        = int(self.MIN_SPEECH_MS       * SAMPLE_RATE / 1000)
        partial_interval  = int(self.PARTIAL_INTERVAL_MS * SAMPLE_RATE / 1000)
        max_chunk         = int(self.MAX_CHUNK_MS         * SAMPLE_RATE / 1000)

        # ── COMMIT: silence detected after enough speech ──────────────────────
        if (samples_since_speech >= silence_threshold and speech_len >= min_speech) \
                or speech_len >= max_chunk:
            commit_chunk = self._buf[self._speech_start: last_end].copy()
            self._buf = self._buf[last_end:]
            self._speech_start    = None
            self._last_partial_at = 0
            return None, commit_chunk   # commit fires, no partial this cycle

        # ── PARTIAL: re-slice speech region at interval ───────────────────────
        partial_chunk = None
        if speech_len >= min_speech and samples_since_partial >= partial_interval:
            partial_chunk = self._buf[self._speech_start: last_end].copy()
            self._last_partial_at = len(self._buf)

        return partial_chunk, None
