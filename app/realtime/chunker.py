import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

_vad_model = load_silero_vad()

# Tunable thresholds
SAMPLE_RATE       = 16000
MIN_SPEECH_MS     = 300          # ignore < 300 ms blips
SILENCE_TRIGGER_MS = 500         # flush after 500 ms of silence
MAX_CHUNK_MS      = 8000         # hard cap – flush anyway after 8 s


class VoiceChunker:
    """
    Stateful chunker that wraps Silero-VAD.

    Call `push(audio_np, sr)` with every incoming frame.
    It returns a completed speech chunk (np.float32 @ 16 kHz)
    when it detects end-of-utterance, otherwise None.
    """

    def __init__(
        self,
        silence_trigger_ms: int = SILENCE_TRIGGER_MS,
        min_speech_ms: int = MIN_SPEECH_MS,
        max_chunk_ms: int = MAX_CHUNK_MS,
    ):
        self.silence_trigger = int(silence_trigger_ms * SAMPLE_RATE / 1000)
        self.min_speech      = int(min_speech_ms      * SAMPLE_RATE / 1000)
        self.max_chunk       = int(max_chunk_ms        * SAMPLE_RATE / 1000)

        self._buf: np.ndarray = np.array([], dtype=np.float32)
        self._speech_start: int | None = None   # sample index inside _buf
        self._silence_since: int = 0            # samples of continuous silence

    # ------------------------------------------------------------------
    def reset(self):
        self._buf = np.array([], dtype=np.float32)
        self._speech_start = None
        self._silence_since = 0

    # ------------------------------------------------------------------
    def push(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """
        Ingest one audio frame.  Returns a completed chunk or None.
        """
        import librosa
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        self._buf = np.concatenate([self._buf, audio])

        # Run VAD on the whole buffer (fast – model is tiny)
        timestamps = _get_timestamps(self._buf)

        if not timestamps:
            # No speech at all – keep a small tail to avoid missing onset
            tail = min(len(self._buf), self.silence_trigger)
            self._buf = self._buf[-tail:]
            self._speech_start  = None
            self._silence_since = 0
            return None

        last_ts = timestamps[-1]
        speech_end_sample = last_ts["end"]

        # Track speech start
        if self._speech_start is None:
            self._speech_start = timestamps[0]["start"]

        samples_after_speech = len(self._buf) - speech_end_sample
        speech_len = speech_end_sample - self._speech_start

        flush = False
        if samples_after_speech >= self.silence_trigger and speech_len >= self.min_speech:
            flush = True  # natural pause
        elif speech_len >= self.max_chunk:
            flush = True  # hard cap

        if flush:
            chunk = self._buf[self._speech_start: speech_end_sample].copy()
            # Keep audio after last speech end (might be onset of next word)
            self._buf = self._buf[speech_end_sample:]
            self._speech_start  = None
            self._silence_since = 0
            return chunk

        return None


# -----------------------------------------------------------------------
def _get_timestamps(audio: np.ndarray):
    t = torch.from_numpy(audio)
    _vad_model.reset_states()
    return get_speech_timestamps(t, _vad_model, sampling_rate=SAMPLE_RATE)
