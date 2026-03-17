import tempfile
import soundfile as sf
import librosa
import numpy as np
import time

from realtime.vad import detect_speech
from services.asr_travel import transcribe_travel
from services.translator import translate
from services.tts_engine import speak


languages = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
    "Urdu": "urd_Arab",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Assamese": "asm_Beng",
    "Sindhi": "snd_Arab",
    "Sanskrit": "san_Deva"
}


def run_travel_pipeline(audio, sr, target_lang):

    start_total = time.time()

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    if len(audio) < 1600:
        return "", "", None, {}

    speech_segments = detect_speech(audio, sr)

    if not speech_segments:
        return "", "", None, {}

    segment = speech_segments[0]
    speech_audio = audio[segment["start"]:segment["end"]]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        sf.write(temp.name, speech_audio, sr)
        temp_path = temp.name

    # ASR
    start_asr = time.time()
    text, src_lang = transcribe_travel(temp_path)
    asr_time = time.time() - start_asr

    if text.strip() == "":
        return "", "", None, {}

    # Translation
    start_trans = time.time()
    translated = translate(text, src_lang, languages[target_lang])
    trans_time = time.time() - start_trans

    # TTS
    start_tts = time.time()
    speech = speak(translated)
    tts_time = time.time() - start_tts

    total_time = time.time() - start_total

    timings = {
        "asr": round(asr_time * 1000, 2),
        "translation": round(trans_time * 1000, 2),
        "tts": round(tts_time * 1000, 2),
        "total": round(total_time * 1000, 2),
    }

    return text, translated, speech, timings
