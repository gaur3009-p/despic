import threading
from realtime.pipeline_state import audio_queue, asr_queue
from services.asr_travel import transcribe_travel


def asr_loop():
    while True:
        chunk = audio_queue.get()

        try:
            text, lang = transcribe_travel(chunk)

            if text.strip():
                asr_queue.put((text, lang))

        except Exception as e:
            print("ASR error:", e)


def start_asr_worker():
    threading.Thread(target=asr_loop, daemon=True).start()
