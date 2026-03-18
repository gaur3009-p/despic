import threading
from realtime.pipeline_state import asr_queue, translation_queue
from services.translator import translate

# Full language map (for safety / fallback if needed)
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


def translation_loop():
    while True:
        # 🔥 Now receiving dynamic target language
        text, src_lang, target_lang = asr_queue.get()

        try:
            # If already code (eng_Latn etc), use directly
            tgt_code = target_lang if "_" in target_lang else LANGUAGES.get(target_lang, "hin_Deva")

            translated = translate(text, src_lang, tgt_code)

            translation_queue.put((text, translated))

        except Exception as e:
            print("Translation error:", e)


def start_translation_worker():
    threading.Thread(target=translation_loop, daemon=True).start()
