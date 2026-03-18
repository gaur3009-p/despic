from __future__ import annotations

import tempfile
import soundfile as sf
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any


# One pool shared across both pipelines.
# max_workers=4 lets up to 4 chunks run ASR+Translate simultaneously.
# Increase if you have more CPU/GPU threads available.
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chunk_worker")


class ParallelChunkProcessor:
    """
    Submit chunks for parallel inference and drain completed results in order.

    Parameters
    ----------
    process_fn : callable
        Function that takes (wav_path: str) and returns any result dict/tuple.
        Runs inside a thread-pool thread — must be thread-safe (model inference
        with GIL-released C extensions like faster-whisper and PyTorch is fine).
    max_in_flight : int
        Back-pressure limit.  If this many chunks are already running, push()
        blocks the VAD callback briefly rather than spawning unbounded futures.
        Set to None to disable back-pressure (default: 8).
    """

    def __init__(
        self,
        process_fn: Callable[[str], Any],
        max_in_flight: int | None = 8,
    ):
        self._fn = process_fn
        self._max = max_in_flight
        # deque of (sequence_id, Future)
        self._queue: deque[tuple[int, Future]] = deque()
        self._seq = 0

    def reset(self):
        """Cancel pending futures and reset sequence counter."""
        while self._queue:
            _, fut = self._queue.popleft()
            fut.cancel()
        self._seq = 0

    def push(self, audio_chunk, sample_rate: int = 16000) -> None:
        """
        Write chunk to a temp WAV and submit to the pool.
        Returns immediately — inference happens in a background thread.
        """
        # Back-pressure: wait until at least one slot frees up
        if self._max is not None:
            while len(self._queue) >= self._max:
                if self._queue[0][1].done():
                    break
                time.sleep(0.005)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_chunk, sample_rate)
            path = tmp.name

        seq = self._seq
        self._seq += 1
        fut = _POOL.submit(self._fn, path)
        self._queue.append((seq, fut))

    def drain(self) -> list[tuple[int, Any]]:
        """
        Return all *completed* futures in sequence order, consuming them.

        Futures that are still running are left in the queue.
        Only leading completed futures are drained — if chunk 2 finishes
        before chunk 1, we wait for chunk 1 first to keep output ordered.
        This is the same ordering guarantee Transformers provide via
        positional encodings + causal masking.
        """
        results = []
        while self._queue:
            seq, fut = self._queue[0]
            if not fut.done():
                break   # head not done — stop to preserve order
            self._queue.popleft()
            exc = fut.exception()
            if exc is not None:
                # Log but don't crash the callback
                print(f"[ParallelChunkProcessor] chunk {seq} failed: {exc}")
                continue
            results.append((seq, fut.result()))
        return results

    @property
    def in_flight(self) -> int:
        return len(self._queue)
