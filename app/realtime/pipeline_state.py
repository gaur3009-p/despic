from queue import Queue

# Pipeline queues
audio_queue = Queue()
asr_queue = Queue()
translation_queue = Queue()

# Buffers (single-user; for multi-user use dict keyed by session_id)
transcript_buffer = ""
translation_buffer = ""
